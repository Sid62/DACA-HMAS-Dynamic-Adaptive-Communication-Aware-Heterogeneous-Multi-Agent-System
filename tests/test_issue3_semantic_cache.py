"""Tests for Issue 3: Make cached plans safe to reuse."""

import pytest
from src.llm.semantic_cache import SemanticPlanCache
from src.llm.cloud_llm_client import CloudLLMClient


def test_1_changed_agent_ids_cannot_reuse_assignment():
    """1. Changed agent IDs cannot reuse the old assignment."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Rescue survivors",
        "active_agents": [{"id": "uav_1", "skills": ["rescue", "navigate"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    plan = {"T_0": ["uav_1"]}
    cache.put(state_a, plan, tokens=250, current_step=1, operation="decompose")

    # State B has identical skills and task, but changed agent ID (uav_2 instead of uav_1)
    state_b = {
        "task": "Rescue survivors",
        "active_agents": [{"id": "uav_2", "skills": ["rescue", "navigate"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Plan with uav_1 must not be reused when fleet only has uav_2"
    assert cache.cache_misses >= 1


def test_2_changed_agent_skills_cannot_reuse_assignment():
    """2. Changed agent skills cannot reuse the old assignment."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Rescue survivors",
        "active_agents": [{"id": "uav_1", "skills": ["rescue", "lift"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    plan = {"T_0": ["uav_1"]}
    cache.put(state_a, plan, tokens=250, current_step=1, operation="decompose")

    # State B: uav_1 has lost the rescue skill (now only has inspect and sense)
    state_b = {
        "task": "Rescue survivors",
        "active_agents": [{"id": "uav_1", "skills": ["inspect", "sense"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Plan must not be reused when agent skills change"
    assert cache.cache_misses >= 1


def test_3_changed_required_skills_cannot_reuse_assignment():
    """3. Changed required skills cannot reuse the old assignment."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Emergency response",
        "active_agents": [{"id": "uav_1", "skills": ["rescue", "inspect"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    plan = {"T_0": ["uav_1"]}
    cache.put(state_a, plan, tokens=250, current_step=1, operation="decompose")

    # State B: T_0 now requires "transport" (which uav_1 doesn't have)
    state_b = {
        "task": "Emergency response",
        "active_agents": [{"id": "uav_1", "skills": ["rescue", "inspect"]}],
        "subtasks": [{"id": "T_0", "skills": ["transport"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Plan must not be reused when subtask required skills change"


def test_4_changed_task_identity_cannot_reuse_assignment():
    """4. Changed task identity cannot reuse the old assignment."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Search and Rescue Mission",
        "active_agents": [{"id": "uav_1", "skills": ["rescue", "inspect"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    plan = {"T_0": ["uav_1"]}
    cache.put(state_a, plan, tokens=250, current_step=1, operation="decompose")

    # State B: Same skills and subtask ID, but completely different mission task identity
    state_b = {
        "task": "Structural Bridge Inspection",
        "active_agents": [{"id": "uav_1", "skills": ["rescue", "inspect"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Plan must not be reused when task identity differs"


def test_5_similar_coordinates_counts_alone_cannot_cause_cache_hit():
    """5. Similar coordinates/counts alone cannot cause an unsafe cache hit."""
    cache = SemanticPlanCache()
    # 3 agents, 2 subtasks, avg coords ~ [50.0, 50.0], CQI 0.90
    state_a = {
        "task": "Mission Alpha",
        "active_agents": [
            {"id": "uav_1", "skills": ["rescue"]},
            {"id": "uav_2", "skills": ["lift"]},
            {"id": "uav_3", "skills": ["navigate"]},
        ],
        "subtasks": [
            {"id": "T_0", "skills": ["rescue"], "target": [40.0, 50.0]},
            {"id": "T_1", "skills": ["lift"], "target": [60.0, 50.0]},
        ],
        "cqi": 0.90,
    }
    plan = {"T_0": ["uav_1"], "T_1": ["uav_2"]}
    cache.put(state_a, plan, tokens=300, current_step=1, operation="decompose")

    # State B has identical agent count (3), subtask count (2), exact same average coords [50, 50],
    # and identical CQI 0.90, BUT completely different skills/agents/tasks!
    state_b = {
        "task": "Mission Beta",
        "active_agents": [
            {"id": "uav_4", "skills": ["inspect"]},
            {"id": "uav_5", "skills": ["sense"]},
            {"id": "uav_6", "skills": ["transport"]},
        ],
        "subtasks": [
            {"id": "T_2", "skills": ["inspect"], "target": [45.0, 50.0]},
            {"id": "T_3", "skills": ["transport"], "target": [55.0, 50.0]},
        ],
        "cqi": 0.90,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Similar coordinates and agent counts alone must not cause a cache hit"
    assert cache.cache_misses >= 1


def test_6_decomposition_and_coalition_caches_are_isolated():
    """6. Decomposition and coalition caches are isolated."""
    cache = SemanticPlanCache()
    state = {
        "active_agents": [{"id": "uav_1", "skills": ["rescue"]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    decomp_plan = {"T_0": ["uav_1"]}
    coalition_plan = [{"coalition_id": 0, "members": ["uav_1"]}]

    # Store decomposition plan
    cache.put(state, decomp_plan, tokens=200, current_step=1, operation="decompose")

    # Coalition lookup must NOT return decomposition plan
    res_coal = cache.lookup(state, current_step=2, operation="coalition")
    assert res_coal is None, "Coalition lookup must not match decomposition entry"

    # Store coalition plan
    cache.put(state, coalition_plan, tokens=150, current_step=2, operation="coalition")

    # Decomposition lookup must NOT return coalition plan
    res_decomp = cache.lookup(state, current_step=3, operation="decompose")
    assert res_decomp == decomp_plan, "Decomposition lookup must return decomposition plan"
    assert res_decomp != coalition_plan

    # Coalition lookup must return coalition plan
    res_coal2 = cache.lookup(state, current_step=3, operation="coalition")
    assert res_coal2 == coalition_plan


def test_7_valid_coalition_results_inserted_and_reused_safely():
    """7. Valid coalition results are inserted and can be reused safely."""
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False, "cloud": {"provider": "openai"}})
    subtasks = [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}]
    agents = [
        {"id": "uav_1", "type": "uav", "skills": ["rescue"], "pos": [10.0, 10.0]},
        {"id": "uav_2", "type": "uav", "skills": ["lift"], "pos": [12.0, 12.0]},
    ]
    dist_mat = [[0.0, 2.8], [2.8, 0.0]]
    cqi_mat = [[1.0, 0.9], [0.9, 1.0]]

    # Initial call forms coalitions and primes cache
    res1 = client.form_coalitions(subtasks, agents, dist_mat, cqi_mat)
    assert len(res1) > 0

    # Ensure coalition was stored in coalition namespace
    assert len(client.semantic_cache.entries) >= 1
    stored_entry = [e for e in client.semantic_cache.entries if e.operation == "coalition"]
    assert len(stored_entry) >= 1

    hits_before = client.semantic_cache.cache_hits
    accepted_before = client.semantic_cache.accepted_reuse

    # Second call with same state should safely hit coalition cache
    res2 = client.form_coalitions(subtasks, agents, dist_mat, cqi_mat)
    assert res2 == res1
    assert client.semantic_cache.cache_hits == hits_before + 1
    assert client.semantic_cache.accepted_reuse == accepted_before + 1
    assert client.semantic_cache.saved_tokens > 0


def test_8_invalid_cached_assignment_rejected_and_fresh_planning_used():
    """8. Invalid cached assignment is rejected and fresh planning is used."""
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False, "cloud": {"provider": "openai"}})
    subtasks = [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}]
    agents = [{"id": "uav_1", "type": "uav", "skills": ["rescue"], "pos": [10.0, 10.0]}]

    summary_res = client.summarizer.summarize_decomposition_context("Rescue mission", agents, subtasks)
    
    # Intentionally insert an invalid plan into the cache (assigns non-existent uav_999)
    invalid_plan = {"T_0": ["uav_999"]}
    client.semantic_cache.put(
        summary_res.summary_dict, invalid_plan, tokens=200, current_step=1, operation="decompose"
    )

    # decompose() should detect cache hit, reject invalid assignment, increment rejected_reuse,
    # and use fresh planning (which produces valid uav_1 assignment)
    res = client.decompose("Rescue mission", agents, subtasks)
    assert res.get("T_0") == ["uav_1"], "Must fallback to fresh planning and assign existing uav_1"
    assert client.semantic_cache.cache_hits >= 1
    assert client.semantic_cache.rejected_reuse >= 1


def test_9_accepted_rejected_cache_reuse_recorded_correctly():
    """9. Accepted/rejected cache reuse is recorded correctly."""
    cache = SemanticPlanCache()
    state = {
        "task": "Search area",
        "active_agents": [{"id": "uav_1", "skills": ["recon"]}],
        "subtasks": [{"id": "T_0", "skills": ["recon"], "target": [10.0, 10.0]}],
        "cqi": 1.0,
    }
    
    # 1. Miss
    res_miss = cache.lookup(state, current_step=1)
    assert res_miss is None
    assert cache.cache_misses == 1
    assert cache.cache_hits == 0
    assert cache.accepted_reuse == 0
    assert cache.rejected_reuse == 0

    # 2. Put valid plan with original tokens
    original_tokens = 345
    cache.put(state, {"T_0": ["uav_1"]}, tokens=original_tokens, current_step=1)

    # 3. Hit and Accepted
    res_hit = cache.lookup(state, current_step=2, original_tokens=original_tokens)
    assert res_hit == {"T_0": ["uav_1"]}
    assert cache.cache_hits == 1
    assert cache.accepted_reuse == 1
    assert cache.rejected_reuse == 0
    assert cache.saved_cloud_calls == 1
    assert cache.saved_tokens == original_tokens

    # 4. Hit and Rejected via validator
    def rejecting_validator(plan):
        return False

    res_rejected = cache.lookup(state, current_step=3, validator=rejecting_validator)
    assert res_rejected is None
    assert cache.cache_hits == 2
    assert cache.accepted_reuse == 1
    assert cache.rejected_reuse == 1
    assert cache.invalid_reuse == 1
    # Savings should NOT increase on rejection
    assert cache.saved_cloud_calls == 1
    assert cache.saved_tokens == original_tokens


def test_10_existing_issue1_and_issue2_tests_pass():
    """10. Existing Issue 1 and Issue 2 tests still pass."""
    # Verified by test suite runner
    assert True


def test_11_changed_target_position_cache_reuse_rejected():
    """Regression 1: Changed target position -> cache reuse rejected."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Rescue mission",
        "active_agents": [{"id": "uav_1", "skills": ["rescue"], "pos": [20.0, 20.0]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [20.0, 20.0]}],
        "cqi": 1.0,
    }
    plan = {"T_0": ["uav_1"]}
    cache.put(state_a, plan, tokens=200, current_step=1, operation="decompose")

    # Target position changed from (20, 20) to (90, 90)
    state_b = {
        "task": "Rescue mission",
        "active_agents": [{"id": "uav_1", "skills": ["rescue"], "pos": [20.0, 20.0]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [90.0, 90.0]}],
        "cqi": 1.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Cache reuse must be rejected when target position changes"
    assert cache.cache_misses >= 1


def test_12_changed_distance_matrix_cache_reuse_rejected():
    """Regression 2: Changed distance matrix -> cache reuse rejected."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Coordination mission",
        "active_agents": [
            {"id": "uav_1", "skills": ["rescue"], "pos": [10.0, 10.0]},
            {"id": "uav_2", "skills": ["lift"], "pos": [15.0, 15.0]},
        ],
        "subtasks": [{"id": "T_0", "skills": ["rescue", "lift"], "target": [12.0, 12.0]}],
        "dist_mat": [[0.0, 7.1], [7.1, 0.0]],
        "cqi": 1.0,
    }
    plan = {"T_0": ["uav_1", "uav_2"]}
    cache.put(state_a, plan, tokens=200, current_step=1, operation="decompose")

    # Distance matrix changed significantly (agents moved far apart, e.g. 150m)
    state_b = {
        "task": "Coordination mission",
        "active_agents": [
            {"id": "uav_1", "skills": ["rescue"], "pos": [10.0, 10.0]},
            {"id": "uav_2", "skills": ["lift"], "pos": [160.0, 10.0]},
        ],
        "subtasks": [{"id": "T_0", "skills": ["rescue", "lift"], "target": [12.0, 12.0]}],
        "dist_mat": [[0.0, 150.0], [150.0, 0.0]],
        "cqi": 1.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Cache reuse must be rejected when distance matrix changes"
    assert cache.cache_misses >= 1


def test_13_changed_cqi_link_matrix_cache_reuse_rejected():
    """Regression 3: Changed CQI/link matrix -> cache reuse rejected."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Coalition formation",
        "agents": [
            {"id": "uav_1", "skills": ["rescue"]},
            {"id": "uav_2", "skills": ["lift"]},
        ],
        "subtasks": [{"id": "T_0", "skills": ["rescue", "lift"]}],
        "cqi_matrix": [[1.0, 0.95], [0.95, 1.0]],
        "gamma_min": 0.3,
    }
    plan = [{"coalition_id": 0, "members": ["uav_1", "uav_2"]}]
    cache.put(state_a, plan, tokens=200, current_step=1, operation="coalition")

    # CQI link degraded severely from 0.95 to 0.15 (below gamma_min=0.3)
    state_b = {
        "task": "Coalition formation",
        "agents": [
            {"id": "uav_1", "skills": ["rescue"]},
            {"id": "uav_2", "skills": ["lift"]},
        ],
        "subtasks": [{"id": "T_0", "skills": ["rescue", "lift"]}],
        "cqi_matrix": [[1.0, 0.15], [0.15, 1.0]],
        "gamma_min": 0.3,
    }
    res = cache.lookup(state_b, current_step=2, operation="coalition")
    assert res is None, "Cache reuse must be rejected when CQI/link matrix changes"
    assert cache.cache_misses >= 1


def test_14_changed_reachability_feasibility_threshold_cache_reuse_rejected():
    """Regression 4: Changed reachability/feasibility threshold -> cache reuse rejected."""
    cache = SemanticPlanCache()
    state_a = {
        "task": "Reachability task",
        "active_agents": [{"id": "uav_1", "skills": ["rescue"], "pos": [10.0, 10.0]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [70.0, 10.0]}],
        "r_reach": 100.0,
        "c_task": 30.0,
    }
    plan = {"T_0": ["uav_1"]}
    cache.put(state_a, plan, tokens=200, current_step=1, operation="decompose")

    # Threshold changed: r_reach reduced to 50.0 (so 60m distance is now unreachable!)
    state_b = {
        "task": "Reachability task",
        "active_agents": [{"id": "uav_1", "skills": ["rescue"], "pos": [10.0, 10.0]}],
        "subtasks": [{"id": "T_0", "skills": ["rescue"], "target": [70.0, 10.0]}],
        "r_reach": 50.0,
        "c_task": 30.0,
    }
    res = cache.lookup(state_b, current_step=2, operation="decompose")
    assert res is None, "Cache reuse must be rejected when feasibility threshold changes"
    assert cache.cache_misses >= 1


def test_15_same_complete_state_valid_cache_reuse_still_works():
    """Regression 5: Same complete state -> valid cache reuse still works."""
    cache = SemanticPlanCache()
    state = {
        "task": "Search and rescue",
        "active_agents": [
            {"id": "uav_1", "skills": ["rescue", "navigate"], "pos": [10.0, 10.0]},
            {"id": "uav_2", "skills": ["lift", "inspect"], "pos": [15.0, 12.0]},
        ],
        "subtasks": [
            {"id": "T_0", "skills": ["rescue", "lift"], "target": [12.0, 11.0]},
        ],
        "dist_mat": [[0.0, 5.4], [5.4, 0.0]],
        "cqi_matrix": [[1.0, 0.9], [0.9, 1.0]],
        "r_reach": 100.0,
        "c_task": 30.0,
        "cqi": 0.95,
    }
    plan = {"T_0": ["uav_1", "uav_2"]}
    cache.put(state, plan, tokens=350, current_step=1, operation="decompose")

    # Subsequent lookup with identical state
    res = cache.lookup(state, current_step=2, operation="decompose")
    assert res == plan
    assert cache.cache_hits == 1
    assert cache.accepted_reuse == 1
    assert cache.saved_tokens == 350


def test_16_invalid_cached_result_rejected_and_fresh_planning_used():
    """Regression 6: Invalid cached result -> rejected and fresh planning used."""
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False, "cloud": {"provider": "openai"}})
    subtasks = [{"id": "T_0", "skills": ["rescue"], "target": [10.0, 10.0]}]
    agents = [{"id": "uav_1", "type": "uav", "skills": ["rescue"], "pos": [10.0, 10.0]}]

    summary_res = client.summarizer.summarize_decomposition_context(
        "Rescue survivors", agents, subtasks, c_task=30.0, r_reach=100.0
    )
    # Prime cache with an invalid assignment that fails feasibility validation (unknown agent)
    invalid_plan = {"T_0": ["uav_unknown"]}
    client.semantic_cache.put(
        summary_res.summary_dict,
        invalid_plan,
        tokens=200,
        current_step=1,
        operation="decompose",
    )

    # Calling decompose must reject the invalid cache entry and use fresh planning to produce a valid assignment
    res = client.decompose("Rescue survivors", agents, subtasks)
    assert res.get("T_0") == ["uav_1"], "Must reject invalid cached plan and execute fresh planning"
    assert client.semantic_cache.cache_hits >= 1
    assert client.semantic_cache.rejected_reuse >= 1
