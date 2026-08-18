"""Tests for state snapshot handoff and verification (Eqs 28-29)."""

import pytest

from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.scenarios import Subtask
from src.handoff.snapshot import (
    AgentSnapshot,
    GlobalSnapshot,
    capture_snapshot,
    restore_snapshot,
    verify_task_preservation,
)


@pytest.fixture
def fleet():
    agents = [
        AgentState(
            "uav_0",
            AgentType.UAV,
            Position(10.0, 20.0, 5.0),
            skills=["navigate"],
            assigned_subtasks=["T_0"],
            completed_subtasks=[],
            remaining_waypoints=[Position(15.0, 25.0, 5.0)],
            coalition_id=0,
        ),
        AgentState(
            "robot_0",
            AgentType.ROBOT,
            Position(30.0, 40.0, 0.0),
            skills=["lift"],
            assigned_subtasks=[],
            completed_subtasks=["T_1"],
            remaining_waypoints=[],
            coalition_id=1,
        ),
    ]
    kin = {
        "uav": KinematicsConfig(15, 1.5),
        "vehicle": KinematicsConfig(10, 0.8),
        "robot": KinematicsConfig(3, 2.0),
    }
    return AgentFleet(agents, kin)


@pytest.fixture
def subtasks():
    return [
        Subtask("T_0", "task 0", Position(50, 50), ["navigate"], completed=False),
        Subtask("T_1", "task 1", Position(60, 60), ["lift"], completed=True),
    ]


@pytest.fixture
def coalitions():
    return [
        {"coalition_id": 0, "members": ["uav_0"], "assigned_tasks": ["T_0"]},
        {"coalition_id": 1, "members": ["robot_0"], "assigned_tasks": ["T_1"]},
    ]


def test_snapshot_restore(fleet, subtasks, coalitions):
    """TEST 1: Capture BEFORE snapshot, mutate runtime state, restore, capture fresh AFTER snapshot, verify equality."""
    before = capture_snapshot(fleet, subtasks, coalitions, 10, 0, 1)

    # Mutate runtime state heavily
    fleet.agents[0].position = Position(99.0, 99.0, 99.0)
    fleet.agents[0].assigned_subtasks = ["T_WRONG"]
    fleet.agents[0].coalition_id = 999
    fleet.agents[1].position = Position(88.0, 88.0, 88.0)

    # Restore state from snapshot
    restore_snapshot(fleet, before)

    # Capture fresh state after restoration
    after = capture_snapshot(fleet, subtasks, coalitions, 10, 0, 1)

    # Verify state preservation between BEFORE snapshot and fresh AFTER state
    assert verify_task_preservation(before, after)
    assert fleet.agents[0].position.x == pytest.approx(10.0)
    assert fleet.agents[0].assigned_subtasks == ["T_0"]
    assert fleet.agents[0].coalition_id == 0


def test_task_preservation(fleet, subtasks, coalitions):
    """TEST 2: Explicitly verify completed subtasks, pending subtasks, and current assignments match."""
    before = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    # Mutate runtime task allocations
    fleet.agents[0].assigned_subtasks = []
    fleet.agents[0].completed_subtasks = ["T_0"]

    # Restore
    restore_snapshot(fleet, before)
    after = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    assert set(before.completed_subtasks) == set(after.completed_subtasks)
    before_pending = set(before.subtask_ids) - set(before.completed_subtasks)
    after_pending = set(after.subtask_ids) - set(after.completed_subtasks)
    assert before_pending == after_pending

    for b_agent, a_agent in zip(before.agents, after.agents):
        assert b_agent.assigned_subtasks == a_agent.assigned_subtasks
        assert b_agent.completed_subtasks == a_agent.completed_subtasks

    assert verify_task_preservation(before, after)


def test_agent_state_preservation(fleet, subtasks, coalitions):
    """TEST 3: Verify agent positions, remaining waypoints, coalition IDs, and state fields."""
    before = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    # Mutate agent kinematics & waypoints
    fleet.agents[0].position = Position(0, 0, 0)
    fleet.agents[0].remaining_waypoints = []
    fleet.agents[0].coalition_id = None

    restore_snapshot(fleet, before)
    after = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    assert after.agents[0].position == pytest.approx([10.0, 20.0, 5.0])
    assert after.agents[0].remaining_waypoints == [[15.0, 25.0, 5.0]]
    assert after.agents[0].coalition_id == 0
    assert verify_task_preservation(before, after)


def test_coalition_preservation(fleet, subtasks, coalitions):
    """TEST 4: Verify coalition structure and task-to-agent mapping preservation."""
    before = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    fleet.agents[0].coalition_id = 1
    fleet.agents[1].coalition_id = 0

    restore_snapshot(fleet, before)
    after = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    assert before.coalitions == after.coalitions
    assert after.agents[0].coalition_id == 0
    assert after.agents[1].coalition_id == 1
    assert verify_task_preservation(before, after)


def test_negative_corruption_test(fleet, subtasks, coalitions):
    """TEST 5: Ensure verification fails when mission-critical fields are corrupted after restoration."""
    before = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)
    restore_snapshot(fleet, before)

    # 1. Corrupt completed subtasks
    after_corrupt_tasks = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)
    after_corrupt_tasks.completed_subtasks = ["T_0", "T_1"]
    assert not verify_task_preservation(before, after_corrupt_tasks)

    # 2. Corrupt assigned subtasks
    after_corrupt_assign = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)
    after_corrupt_assign.agents[0].assigned_subtasks = ["T_CORRUPTED"]
    assert not verify_task_preservation(before, after_corrupt_assign)

    # 3. Corrupt agent position
    after_corrupt_pos = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)
    after_corrupt_pos.agents[0].position = [999.0, 999.0, 999.0]
    assert not verify_task_preservation(before, after_corrupt_pos)

    # 4. Corrupt coalition ID
    after_corrupt_coalition = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)
    after_corrupt_coalition.agents[0].coalition_id = 999
    assert not verify_task_preservation(before, after_corrupt_coalition)

    # 5. Corrupt coalition list
    after_corrupt_coal_list = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)
    after_corrupt_coal_list.coalitions = [{"coalition_id": 99, "members": ["uav_0"]}]
    assert not verify_task_preservation(before, after_corrupt_coal_list)


def test_continue_execution(fleet, subtasks, coalitions):
    """TEST 6: Mission continuity — restore state and ensure pending task T_0 can be executed/completed."""
    # T_0 is pending, T_1 is completed
    before = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    # Simulate architecture switch state disruption
    fleet.agents[0].position = Position(0, 0, 0)
    fleet.agents[0].assigned_subtasks = []

    # Restore state
    restore_snapshot(fleet, before)
    after = capture_snapshot(fleet, subtasks, coalitions, 5, 0, 1)

    # Verify state preservation
    assert verify_task_preservation(before, after)
    assert not subtasks[0].completed
    assert subtasks[1].completed

    # Continue mission execution on restored state: complete pending task T_0
    subtasks[0].completed = True
    fleet.agents[0].assigned_subtasks = []
    fleet.agents[0].completed_subtasks.append("T_0")

    # Post-execution check: both tasks now completed
    assert subtasks[0].completed
    assert subtasks[1].completed
    assert "T_0" in fleet.agents[0].completed_subtasks
    assert "T_1" in fleet.agents[1].completed_subtasks


def test_plan_state_known_mode_tracking_across_architecture_switch(fleet, subtasks, coalitions):
    """TEST 7: Mode tracking consistency — verify known_mode updates when continuity absorbs switch."""
    from src.coordination.replan_trigger import PlanState, should_replan, update_plan_state
    from src.coordination.plan_continuity import PlanContinuityEngine

    assignments = {"T_0": ["uav_0"]}
    plan_state = PlanState()
    update_plan_state(plan_state, subtasks, fleet, coalitions, assignments, mode=0)

    cont_engine = PlanContinuityEngine()
    cont_engine.set_active_plan(assignments, coalitions, subtasks, mode=0)

    assert plan_state.known_mode == 0

    # Step 1: Switch to mode 1 (Decentralized) with valid plan -> continuity absorbs it
    replan_now, reason = should_replan(
        plan_state, subtasks, fleet, coalitions, mode=1, continuity_engine=cont_engine
    )
    assert not replan_now
    assert plan_state.known_mode == 1, "known_mode must update to 1 when continuity absorbs switch"

    # Step 2: Next step in mode 1 -> should NOT re-trigger architecture_switched
    replan_now_2, reason_2 = should_replan(
        plan_state, subtasks, fleet, coalitions, mode=1, continuity_engine=cont_engine
    )
    assert not replan_now_2
    assert reason_2 == ""
    assert plan_state.known_mode == 1


def test_distributed_state_restoration_no_duplication(fleet, subtasks, coalitions):
    """TEST 8: Distributed state restoration — verify inboxes and node_states do not duplicate messages."""
    from src.communication.peer_manager import PeerCommunicationManager
    from src.handoff.snapshot import capture_snapshot, restore_distributed_state, verify_task_preservation
    from src.llm.device_llm_client import DeviceLLMClient

    pm = PeerCommunicationManager()
    pm.register_domain_peers(["uav", "robot", "vehicle"])
    pm.send_message("uav", "robot", "realloc_proposal", {"test": 1})

    device_llms = {
        "uav": DeviceLLMClient(node_id="uav"),
        "robot": DeviceLLMClient(node_id="robot"),
        "vehicle": DeviceLLMClient(node_id="vehicle"),
    }

    # Capture BEFORE snapshot with 1 in-flight message in robot's inbox
    before = capture_snapshot(
        fleet, subtasks, coalitions, 10, 0, 1,
        device_llms=device_llms,
        pending_messages=pm.pending_messages_all(),
    )
    assert len(before.pending_messages.get("robot", [])) == 1

    # Restore distributed state
    restore_distributed_state(before, device_llms, peer_manager=pm)

    # Capture AFTER snapshot
    after = capture_snapshot(
        fleet, subtasks, coalitions, 10, 0, 1,
        device_llms=device_llms,
        pending_messages=pm.pending_messages_all(),
    )

    # Verification must pass: inboxes must not have duplicated
    assert len(after.pending_messages.get("robot", [])) == 1
    assert verify_task_preservation(before, after)


def test_post_switch_peer_communication_after_empty_restore(fleet, subtasks, coalitions):
    """TEST 9: Post-switch peer broadcast & message delivery after empty state handoff restore."""
    from src.communication.peer_manager import PeerCommunicationManager
    from src.handoff.snapshot import capture_snapshot, restore_distributed_state
    from src.llm.device_llm_client import DeviceLLMClient

    pm = PeerCommunicationManager()
    domains = ["uav", "robot", "vehicle"]
    pm.register_domain_peers(domains)

    device_llms = {d: DeviceLLMClient(node_id=d) for d in domains}

    # Empty snapshot before handoff (no pending messages)
    before = capture_snapshot(
        fleet, subtasks, coalitions, 39, 0, 1,
        device_llms=device_llms,
        pending_messages=pm.pending_messages_all(),
    )
    assert before.pending_messages == {}

    # State restoration
    restore_distributed_state(before, device_llms, peer_manager=pm)

    # Post-switch broadcast across all registered domains
    for sender in domains:
        delivered = pm.broadcast(sender, "realloc_proposal", {"test": sender})
        assert delivered == len(domains) - 1, f"Broadcast from {sender} should deliver to {len(domains)-1} peers"

    # All receivers can receive their messages cleanly without KeyError
    for d in domains:
        msgs = pm.receive_messages(d)
        assert len(msgs) == len(domains) - 1



