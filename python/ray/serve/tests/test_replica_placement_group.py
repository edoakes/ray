import sys

import pytest

import ray
from ray.util.placement_group import (
    get_current_placement_group,
    PlacementGroup,
)
from ray._private.test_utils import wait_for_condition

from ray import serve
from ray.serve.context import get_global_client
from ray.serve._private.utils import get_all_live_placement_group_names


def _get_pg_strategy(pg: PlacementGroup) -> str:
    return ray.util.placement_group_table(pg)["strategy"]


def test_basic(serve_instance):
    """Test the basic workflow: multiple replicas with their own PGs."""

    @serve.deployment(
        num_replicas=2,
        placement_group_bundles=[{"CPU": 1}, {"CPU": 0.1}],
    )
    class D:
        def get_pg(self) -> PlacementGroup:
            return get_current_placement_group()

    h = serve.run(D.bind(), name="pg_test")

    # Verify that each replica has its own placement group with the correct config.
    assert len(get_all_live_placement_group_names()) == 2
    unique_pgs = set(ray.get([h.get_pg.remote() for _ in range(20)]))
    assert len(unique_pgs) == 2
    for pg in unique_pgs:
        assert _get_pg_strategy(pg) == "PACK"
        assert pg.bundle_specs == [{"CPU": 1}, {"CPU": 0.1}]

    # Verify that all placement groups are deleted when the deployment is deleted.
    serve.delete("pg_test")
    assert len(get_all_live_placement_group_names()) == 0


def test_upgrade_and_change_pg(serve_instance):
    """Test re-deploying a deployment with different PG bundles and strategy."""

    @serve.deployment(
        num_replicas=1,
        placement_group_bundles=[{"CPU": 1}, {"CPU": 0.1}],
        placement_group_strategy="STRICT_PACK",
    )
    class D:
        def get_pg(self) -> PlacementGroup:
            return get_current_placement_group()

    h = serve.run(D.bind(), name="pg_test")

    # Check that the original replica is created with the expected PG config.
    assert len(get_all_live_placement_group_names()) == 1
    original_pg = ray.get(h.get_pg.remote())
    assert original_pg.bundle_specs == [{"CPU": 1}, {"CPU": 0.1}]
    assert _get_pg_strategy(original_pg) == "STRICT_PACK"

    # Re-deploy with a new PG config.
    D = D.options(
        placement_group_bundles=[{"CPU": 2}, {"CPU": 0.2}],
        placement_group_strategy="SPREAD",
    )

    h = serve.run(D.bind(), name="pg_test")
    assert len(get_all_live_placement_group_names()) == 1
    new_pg = ray.get(h.get_pg.remote())
    assert new_pg.bundle_specs == [{"CPU": 2}, {"CPU": 0.2}]
    assert _get_pg_strategy(new_pg) == "SPREAD"

    # Verify that all placement groups are deleted when the deployment is deleted.
    serve.delete("pg_test")
    assert len(get_all_live_placement_group_names()) == 0


def test_pg_removed_on_replica_graceful_shutdown(serve_instance):
    """Verify that PGs are removed when a replica shuts down gracefully."""

    @serve.deployment(
        placement_group_bundles=[{"CPU": 1}],
    )
    class D:
        def get_pg(self) -> PlacementGroup:
            return get_current_placement_group()

    h = serve.run(D.options(num_replicas=2).bind(), name="pg_test")

    # Two replicas to start, each should have their own placement group.
    assert len(get_all_live_placement_group_names()) == 2
    original_unique_pgs = set(ray.get([h.get_pg.remote() for _ in range(20)]))
    assert len(original_unique_pgs) == 2

    # Re-deploy the application with a single replica.
    # The existing PGs should be removed and a new one should be created for the
    # new replica.
    h = serve.run(D.options(num_replicas=1).bind(), name="pg_test")
    assert len(get_all_live_placement_group_names()) == 1
    new_unique_pgs = set(ray.get([h.get_pg.remote() for _ in range(20)]))
    assert len(new_unique_pgs) == 1
    assert not new_unique_pgs.issubset(original_unique_pgs)

    # Verify that all placement groups are deleted when the deployment is deleted.
    serve.delete("pg_test")
    assert len(get_all_live_placement_group_names()) == 0


def test_pg_removed_on_replica_crash(serve_instance):
    """Verify that PGs are removed when a replica crashes unexpectedly."""

    @serve.deployment(
        placement_group_bundles=[{"CPU": 1}],
        health_check_period_s=0.1,
    )
    class D:
        def die(self):
            sys.exit()

        def get_pg(self) -> PlacementGroup:
            return get_current_placement_group()

    h = serve.run(D.bind(), name="pg_test")

    # Get the placement group for the original replica.
    assert len(get_all_live_placement_group_names()) == 1
    pg = ray.get(h.get_pg.remote())

    # Kill the replica forcefully.
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(h.die.remote())

    def new_replica_scheduled():
        try:
            ray.get(h.get_pg.remote())
        except ray.exceptions.RayActorError:
            return False

        return True

    # The original placement group should be deleted and a new replica should
    # be scheduled with its own new placement group.
    wait_for_condition(new_replica_scheduled)
    new_pg = ray.get(h.get_pg.remote())
    assert pg != new_pg
    assert len(get_all_live_placement_group_names()) == 1


def test_pg_removed_after_controller_crash(serve_instance):
    """Verify that PGs are removed normally after recovering from a controller crash.

    If the placement group was not properly recovered in the replica recovery process,
    it would be leaked here.
    """

    @serve.deployment(
        placement_group_bundles=[{"CPU": 1}],
    )
    class D:
        pass

    serve.run(D.bind(), name="pg_test")
    assert len(get_all_live_placement_group_names()) == 1

    ray.kill(get_global_client()._controller, no_restart=False)

    serve.delete("pg_test")
    assert len(get_all_live_placement_group_names()) == 0


def test_leaked_pg_removed_on_controller_recovery(serve_instance):
    """Verify that leaked PGs are removed on controller recovery.

    A placement group can be "leaked" if the replica is killed while the controller is
    down or the controller crashes between creating a placement group and its replica.

    In these cases, the controller should detect the leak on recovery and delete the
    leaked placement group(s).
    """

    @serve.deployment(
        placement_group_bundles=[{"CPU": 1}],
        health_check_period_s=0.1,
    )
    class D:
        def die(self):
            sys.exit()

        def get_pg(self) -> PlacementGroup:
            return get_current_placement_group()

    h = serve.run(D.bind(), name="pg_test")

    prev_pg = ray.get(h.get_pg.remote())
    assert len(get_all_live_placement_group_names()) == 1

    # Kill the controller and the replica immediately after.
    # This will cause the controller to *not* detect the replica on recovery, but it
    # should still detect the leaked placement group and clean it up.
    ray.kill(get_global_client()._controller, no_restart=False)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(h.die.remote())

    def leaked_pg_cleaned_up():
        try:
            new_pg = ray.get(h.get_pg.remote())
        except ray.exceptions.RayActorError:
            return False

        return len(get_all_live_placement_group_names()) == 1 and new_pg != prev_pg

    # Verify that a new replica is placed with a new placement group and the old
    # placement group has been removed.
    wait_for_condition(leaked_pg_cleaned_up)

    serve.delete("pg_test")
    assert len(get_all_live_placement_group_names()) == 0


def test_replica_actor_infeasible(serve_instance):
    """Test that we get a validation error if the replica doesn't fit in the bundle."""

    @serve.deployment(
        placement_group_bundles=[{"CPU": 0.1}],
    )
    class Infeasible:
        pass

    with pytest.raises(ValueError):
        serve.run(Infeasible.bind())


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
