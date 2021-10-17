import pytest
from pydantic import ValidationError

from ray.serve.config import (BackendConfig, DeploymentMode, HTTPOptions,
                              ReplicaConfig, SerializedFuncOrClass)


def test_backend_config_validation():
    # Test unknown key.
    with pytest.raises(ValidationError):
        BackendConfig(unknown_key=-1)

    # Test num_replicas validation.
    BackendConfig(num_replicas=1)
    with pytest.raises(ValidationError, match="type_error"):
        BackendConfig(num_replicas="hello")
    with pytest.raises(ValidationError, match="value_error"):
        BackendConfig(num_replicas=-1)

    # Test dynamic default for max_concurrent_queries.
    assert BackendConfig().max_concurrent_queries == 100


def test_backend_config_update():
    b = BackendConfig(num_replicas=1, max_concurrent_queries=1)

    # Test updating a key works.
    b.num_replicas = 2
    assert b.num_replicas == 2
    # Check that not specifying a key doesn't update it.
    assert b.max_concurrent_queries == 1

    # Check that input is validated.
    with pytest.raises(ValidationError):
        b.num_replicas = "Hello"
    with pytest.raises(ValidationError):
        b.num_replicas = -1


class TestClass:
    pass


def test_function():
    pass


serialized_class = SerializedFuncOrClass(TestClass)
serialized_function = SerializedFuncOrClass(test_function)


class TestReplicaConfig:
    def test_func_or_class_validation(self):
        r1 = ReplicaConfig(func_or_class=TestClass)
        assert not r1.serialized_func_or_class.is_function
        r1.set_func_or_class(test_function)
        assert r1.serialized_func_or_class.is_function
        r2 = ReplicaConfig(func_or_class=test_function)
        assert r2.serialized_func_or_class.is_function
        r2.set_func_or_class(TestClass)
        assert not r2.serialized_func_or_class.is_function
        r3 = ReplicaConfig(func_or_class=serialized_class)
        assert not r3.serialized_func_or_class.is_function
        r3.set_func_or_class(serialized_function)
        assert r3.serialized_func_or_class.is_function
        r4 = ReplicaConfig(func_or_class=serialized_function)
        assert r4.serialized_func_or_class.is_function
        r4.set_func_or_class(serialized_class)
        assert not r4.serialized_func_or_class.is_function

        with pytest.raises(TypeError):
            ReplicaConfig(func_or_class=TestClass())

    def test_ray_actor_options_conversion(self):
        r = ReplicaConfig(func_or_class=TestClass)
        r.set_ray_actor_options({
            "num_cpus": 1.0,
            "num_gpus": 10,
            "resources": {
                "abc": 1.0
            },
            "memory": 1000000.0,
            "object_store_memory": 1000000,
        })
        assert r.num_cpus == 1.0
        assert r.num_gpus == 10.0
        assert r.resources == {
            "abc": 1.0,
            "memory": 1000000.0,
            "object_store_memory": 1000000
        }

    def test_ray_actor_options_invalid_types(self):
        r = ReplicaConfig(func_or_class=TestClass)
        with pytest.raises(TypeError):
            r.set_ray_actor_options(1.0)
        with pytest.raises(TypeError):
            r.set_ray_actor_options({"num_cpus": "hello"})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"num_cpus": -1})
        with pytest.raises(TypeError):
            r.set_ray_actor_options({"num_gpus": "hello"})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"num_gpus": -1})
        with pytest.raises(TypeError):
            r.set_ray_actor_options({"memory": "hello"})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"memory": -1})
        with pytest.raises(TypeError):
            r.set_ray_actor_options({"object_store_memory": "hello"})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"object_store_memory": -1})
        with pytest.raises(TypeError):
            r.set_ray_actor_options({"resources": None})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"name": None})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"lifetime": None})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"max_restarts": None})
        with pytest.raises(ValueError):
            r.set_ray_actor_options({"placement_group": None})

    def test_num_cpus(self):
        pass

    def test_num_gpus(self):
        pass

    def test_resources(self):
        pass

    def test_accelerator_type(self):
        pass

    def test_runtime_env(self):
        pass


def test_http_options():
    HTTPOptions()
    HTTPOptions(host="8.8.8.8", middlewares=[object()])
    assert HTTPOptions(host=None).location == "NoServer"
    assert HTTPOptions(location=None).location == "NoServer"
    assert HTTPOptions(
        location=DeploymentMode.EveryNode).location == "EveryNode"


def test_with_proto():
    # Test roundtrip
    config = BackendConfig(num_replicas=100, max_concurrent_queries=16)
    assert config == BackendConfig.from_proto_bytes(config.to_proto_bytes())

    # Test user_config object
    config = BackendConfig(user_config={"python": ("native", ["objects"])})
    assert config == BackendConfig.from_proto_bytes(config.to_proto_bytes())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", "-s", __file__]))
