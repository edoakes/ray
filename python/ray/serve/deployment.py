import inspect
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ray.dag.py_obj_scanner import _PyObjScanner
from ray.serve._private.common import DeploymentHandleSource
from ray.serve._private.config import (
    DeploymentConfig,
    ReplicaConfig,
    handle_num_replicas_auto,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import DEFAULT, Default
from ray.serve.config import AutoscalingConfig
from ray.serve.context import _get_global_client
from ray.serve.handle import DeploymentHandle, _HandleOptions
from ray.serve.schema import DeploymentSchema, LoggingConfig, RayActorOptionsSchema
from ray.util.annotations import PublicAPI

logger = logging.getLogger(SERVE_LOGGER_NAME)


@PublicAPI(stability="stable")
class Application:
    """One or more deployments bound with arguments that can be deployed together.

    Can be passed into another `Deployment.bind()` to compose multiple deployments in a
    single application, passed to `serve.run`, or deployed via a Serve config file.

    For example, to define an Application and run it in Python:

        .. code-block:: python

            from ray import serve
            from ray.serve import Application

            @serve.deployment
            class MyDeployment:
                pass

            app: Application = MyDeployment.bind(OtherDeployment.bind())
            serve.run(app)

    To run the same app using the command line interface (CLI):

        .. code-block:: bash

            serve run python_file:app

    To deploy the same app via a config file:

        .. code-block:: yaml

            applications:
                my_app:
                    import_path: python_file:app

    """

    def __init__(self, bound_deployment: "Deployment"):
        self._bound_deployment = bound_deployment

    def _get_deployment_name(
        self, app: "Application", deployment_names: Dict["Application", str]
    ):
        if app in deployment_names:
            return deployment_names[app]

        name = app._bound_deployment.name
        idx = 1
        while name in deployment_names.values():
            name = f"{app._bound_deployment.name}_{idx}"
            idx += 1

        return name

    def _inject_handles_recursive(
        self,
        *,
        app_name: str,
        handles: Dict["Application", DeploymentHandle],
        deployment_names: Dict["Application", str],
    ) -> List["Deployment"]:
        deployments = []
        scanner = _PyObjScanner(source_type=Application)
        try:
            apps = scanner.find_nodes(
                (self._bound_deployment.init_args, self._bound_deployment.init_kwargs)
            )
            for app in apps:
                if app not in handles:
                    deployment_names[app] = self._get_deployment_name(
                        app, deployment_names
                    )
                    handles[app] = DeploymentHandle(
                        deployment_names[app],
                        app_name="default",
                        handle_options=_HandleOptions(
                            _source=DeploymentHandleSource.REPLICA
                        ),
                    )

                deployments.extend(
                    app._inject_handles_recursive(
                        app_name=app_name,
                        handles=handles,
                        deployment_names=deployment_names,
                    )
                )

            new_init_args, new_init_kwargs = scanner.replace_nodes(handles)
            # TODO: update name if necessary.
            deployment_names[self] = self._get_deployment_name(self, deployment_names)
            deployments.append(
                self._bound_deployment.options(
                    name=deployment_names[self],
                    _init_args=new_init_args,
                    _init_kwargs=new_init_kwargs,
                )
            )
            return deployments
        finally:
            scanner.clear()

    def _build(self, *, app_name: str = "default") -> Tuple[str, List["Deployment"]]:
        deployments = self._inject_handles_recursive(
            app_name=app_name,
            handles={},
            deployment_names={},
        )
        return self._bound_deployment.name, deployments

    def __hash__(self):
        return id(self)  # sketchy?


@PublicAPI(stability="stable")
class Deployment:
    """Class (or function) decorated with the `@serve.deployment` decorator.

    This is run on a number of replica actors. Requests to those replicas call
    this class.

    One or more deployments can be composed together into an `Application` which is
    then run via `serve.run` or a config file.

    Example:

    .. code-block:: python

        @serve.deployment
        class MyDeployment:
            def __init__(self, name: str):
                self._name = name

            def __call__(self, request):
                return "Hello world!"

            app = MyDeployment.bind()
            # Run via `serve.run` or the `serve run` CLI command.
            serve.run(app)

    """

    def __init__(
        self,
        name: str,
        deployment_config: DeploymentConfig,
        replica_config: ReplicaConfig,
        version: Optional[str] = None,
        route_prefix: Union[str, None, DEFAULT] = DEFAULT.VALUE,
        _internal=False,
    ) -> None:
        if not _internal:
            raise RuntimeError(
                "The Deployment constructor should not be called "
                "directly. Use `@serve.deployment` instead."
            )
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not (version is None or isinstance(version, str)):
            raise TypeError("version must be a string.")
        if route_prefix is not DEFAULT.VALUE and route_prefix is not None:
            if not isinstance(route_prefix, str):
                raise TypeError("route_prefix must be a string.")
            if not route_prefix.startswith("/"):
                raise ValueError("route_prefix must start with '/'.")
            if route_prefix != "/" and route_prefix.endswith("/"):
                raise ValueError(
                    "route_prefix must not end with '/' unless it's the root."
                )
            if "{" in route_prefix or "}" in route_prefix:
                raise ValueError("route_prefix may not contain wildcards.")

        docs_path = None
        if (
            inspect.isclass(replica_config.deployment_def)
            and hasattr(replica_config.deployment_def, "__module__")
            and replica_config.deployment_def.__module__ == "ray.serve.api"
            and hasattr(replica_config.deployment_def, "__fastapi_docs_path__")
        ):
            docs_path = replica_config.deployment_def.__fastapi_docs_path__

        self._name = name
        self._version = version
        self._deployment_config = deployment_config
        self._replica_config = replica_config
        self._route_prefix = route_prefix
        self._docs_path = docs_path

    @property
    def name(self) -> str:
        """Unique name of this deployment."""
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def func_or_class(self) -> Union[Callable, str]:
        """Underlying class or function that this deployment wraps."""
        return self._replica_config.deployment_def

    @property
    def num_replicas(self) -> int:
        """Target number of replicas."""
        return self._deployment_config.num_replicas

    @property
    def user_config(self) -> Any:
        """Dynamic user-provided config options."""
        return self._deployment_config.user_config

    @property
    def max_ongoing_requests(self) -> int:
        """Max number of requests a replica can handle at once."""
        return self._deployment_config.max_ongoing_requests

    @property
    def max_queued_requests(self) -> int:
        """Max number of requests that can be queued in each deployment handle."""
        return self._deployment_config.max_queued_requests

    @property
    def route_prefix(self) -> Optional[str]:
        """HTTP route prefix that this deployment is exposed under."""
        if self._route_prefix is DEFAULT.VALUE:
            return f"/{self._name}"
        return self._route_prefix

    @property
    def ray_actor_options(self) -> Optional[Dict]:
        """Actor options such as resources required for each replica."""
        return self._replica_config.ray_actor_options

    @property
    def init_args(self) -> Tuple[Any]:
        return self._replica_config.init_args

    @property
    def init_kwargs(self) -> Tuple[Any]:
        return self._replica_config.init_kwargs

    @property
    def url(self) -> Optional[str]:
        if self._route_prefix is None:
            # this deployment is not exposed over HTTP
            return None

        return _get_global_client().root_url + self.route_prefix

    @property
    def logging_config(self) -> Dict:
        return self._deployment_config.logging_config

    def set_logging_config(self, logging_config: Dict):
        self._deployment_config.logging_config = logging_config

    def __call__(self):
        raise RuntimeError(
            "Deployments cannot be constructed directly. "
            "Use `deployment.deploy() instead.`"
        )

    def bind(self, *args, **kwargs) -> Application:
        """Bind the arguments to the deployment and return an Application.

        The returned Application can be deployed using `serve.run` (or via
        config file) or bound to another deployment for composition.
        """
        return Application(self.options(_init_args=args, _init_kwargs=kwargs))

    def _deploy(self, *init_args, _blocking=True, **init_kwargs):
        """Deploy or update this deployment.

        Args:
            init_args: args to pass to the class __init__
                method. Not valid if this deployment wraps a function.
            init_kwargs: kwargs to pass to the class __init__
                method. Not valid if this deployment wraps a function.
        """
        if len(init_args) == 0 and self._replica_config.init_args is not None:
            init_args = self._replica_config.init_args
        if len(init_kwargs) == 0 and self._replica_config.init_kwargs is not None:
            init_kwargs = self._replica_config.init_kwargs

        replica_config = ReplicaConfig.create(
            self._replica_config.deployment_def,
            init_args=init_args,
            init_kwargs=init_kwargs,
            ray_actor_options=self._replica_config.ray_actor_options,
            placement_group_bundles=self._replica_config.placement_group_bundles,
            placement_group_strategy=self._replica_config.placement_group_strategy,
            max_replicas_per_node=self._replica_config.max_replicas_per_node,
        )

        return _get_global_client().deploy(
            self._name,
            replica_config=replica_config,
            deployment_config=self._deployment_config,
            version=self._version,
            route_prefix=self.route_prefix,
            url=self.url,
            _blocking=_blocking,
        )

    def _delete(self):
        """Delete this deployment."""

        return _get_global_client().delete_deployments([self._name])

    def options(
        self,
        func_or_class: Optional[Callable] = None,
        name: Default[str] = DEFAULT.VALUE,
        version: Default[str] = DEFAULT.VALUE,
        num_replicas: Default[Optional[Union[int, str]]] = DEFAULT.VALUE,
        route_prefix: Default[Union[str, None]] = DEFAULT.VALUE,
        ray_actor_options: Default[Optional[Dict]] = DEFAULT.VALUE,
        placement_group_bundles: Default[List[Dict[str, float]]] = DEFAULT.VALUE,
        placement_group_strategy: Default[str] = DEFAULT.VALUE,
        max_replicas_per_node: Default[int] = DEFAULT.VALUE,
        user_config: Default[Optional[Any]] = DEFAULT.VALUE,
        max_ongoing_requests: Default[int] = DEFAULT.VALUE,
        max_queued_requests: Default[int] = DEFAULT.VALUE,
        autoscaling_config: Default[
            Union[Dict, AutoscalingConfig, None]
        ] = DEFAULT.VALUE,
        graceful_shutdown_wait_loop_s: Default[float] = DEFAULT.VALUE,
        graceful_shutdown_timeout_s: Default[float] = DEFAULT.VALUE,
        health_check_period_s: Default[float] = DEFAULT.VALUE,
        health_check_timeout_s: Default[float] = DEFAULT.VALUE,
        logging_config: Default[Union[Dict, LoggingConfig, None]] = DEFAULT.VALUE,
        _init_args: Default[Tuple[Any]] = DEFAULT.VALUE,
        _init_kwargs: Default[Dict[Any, Any]] = DEFAULT.VALUE,
        _internal: bool = False,
    ) -> "Deployment":
        """Return a copy of this deployment with updated options.

        Only those options passed in will be updated, all others will remain
        unchanged from the existing deployment.

        Refer to the `@serve.deployment` decorator docs for available arguments.
        """

        # Modify max_ongoing_requests and autoscaling_config if
        # `num_replicas="auto"`
        if max_ongoing_requests is None:
            raise ValueError("`max_ongoing_requests` must be non-null, got None.")
        if num_replicas == "auto":
            num_replicas = None
            max_ongoing_requests, autoscaling_config = handle_num_replicas_auto(
                max_ongoing_requests, autoscaling_config
            )

            ServeUsageTag.AUTO_NUM_REPLICAS_USED.record("1")

        # NOTE: The user_configured_option_names should be the first thing that's
        # defined in this method. It depends on the locals() dictionary storing
        # only the function args/kwargs.
        # Create list of all user-configured options from keyword args
        user_configured_option_names = [
            option
            for option, value in locals().items()
            if option not in {"self", "func_or_class", "_internal"}
            and value is not DEFAULT.VALUE
        ]

        new_deployment_config = deepcopy(self._deployment_config)
        if not _internal:
            new_deployment_config.user_configured_option_names.update(
                user_configured_option_names
            )

        if num_replicas not in [
            DEFAULT.VALUE,
            None,
            "auto",
        ] and autoscaling_config not in [
            DEFAULT.VALUE,
            None,
        ]:
            raise ValueError(
                "Manually setting num_replicas is not allowed when "
                "autoscaling_config is provided."
            )

        if num_replicas == 0:
            raise ValueError("num_replicas is expected to larger than 0")

        if not _internal and version is not DEFAULT.VALUE:
            logger.warning(
                "DeprecationWarning: `version` in `Deployment.options()` has been "
                "deprecated. Explicitly specifying version will raise an error in the "
                "future!"
            )

        if not _internal and route_prefix is not DEFAULT.VALUE:
            logger.warning(
                "DeprecationWarning: `route_prefix` in `@serve.deployment` has been "
                "deprecated. To specify a route prefix for an application, pass it "
                "into `serve.run` instead."
            )

        elif num_replicas not in [DEFAULT.VALUE, None]:
            new_deployment_config.num_replicas = num_replicas

        if user_config is not DEFAULT.VALUE:
            new_deployment_config.user_config = user_config

        if max_ongoing_requests is not DEFAULT.VALUE:
            new_deployment_config.max_ongoing_requests = max_ongoing_requests

        if max_queued_requests is not DEFAULT.VALUE:
            new_deployment_config.max_queued_requests = max_queued_requests

        if func_or_class is None:
            func_or_class = self._replica_config.deployment_def

        if name is DEFAULT.VALUE:
            name = self._name

        if version is DEFAULT.VALUE:
            version = self._version

        if _init_args is DEFAULT.VALUE:
            _init_args = self._replica_config.init_args

        if _init_kwargs is DEFAULT.VALUE:
            _init_kwargs = self._replica_config.init_kwargs

        if route_prefix is DEFAULT.VALUE:
            # Default is to keep the previous value
            route_prefix = self._route_prefix

        if ray_actor_options is DEFAULT.VALUE:
            ray_actor_options = self._replica_config.ray_actor_options

        if placement_group_bundles is DEFAULT.VALUE:
            placement_group_bundles = self._replica_config.placement_group_bundles

        if placement_group_strategy is DEFAULT.VALUE:
            placement_group_strategy = self._replica_config.placement_group_strategy

        if max_replicas_per_node is DEFAULT.VALUE:
            max_replicas_per_node = self._replica_config.max_replicas_per_node

        if autoscaling_config is not DEFAULT.VALUE:
            new_deployment_config.autoscaling_config = autoscaling_config

        if graceful_shutdown_wait_loop_s is not DEFAULT.VALUE:
            new_deployment_config.graceful_shutdown_wait_loop_s = (
                graceful_shutdown_wait_loop_s
            )

        if graceful_shutdown_timeout_s is not DEFAULT.VALUE:
            new_deployment_config.graceful_shutdown_timeout_s = (
                graceful_shutdown_timeout_s
            )

        if health_check_period_s is not DEFAULT.VALUE:
            new_deployment_config.health_check_period_s = health_check_period_s

        if health_check_timeout_s is not DEFAULT.VALUE:
            new_deployment_config.health_check_timeout_s = health_check_timeout_s

        if logging_config is not DEFAULT.VALUE:
            if isinstance(logging_config, LoggingConfig):
                logging_config = logging_config.dict()
            new_deployment_config.logging_config = logging_config

        new_replica_config = ReplicaConfig.create(
            func_or_class,
            init_args=_init_args,
            init_kwargs=_init_kwargs,
            ray_actor_options=ray_actor_options,
            placement_group_bundles=placement_group_bundles,
            placement_group_strategy=placement_group_strategy,
            max_replicas_per_node=max_replicas_per_node,
        )

        return Deployment(
            name,
            new_deployment_config,
            new_replica_config,
            version=version,
            route_prefix=route_prefix,
            _internal=True,
        )

    def __eq__(self, other):
        return all(
            [
                self._name == other._name,
                self._version == other._version,
                self._deployment_config == other._deployment_config,
                self._replica_config.init_args == other._replica_config.init_args,
                self._replica_config.init_kwargs == other._replica_config.init_kwargs,
                # compare route prefix with default value resolved
                self.route_prefix == other.route_prefix,
                self._replica_config.ray_actor_options
                == self._replica_config.ray_actor_options,
            ]
        )

    def __str__(self):
        return (
            f"Deployment(name={self._name},"
            f"version={self._version},"
            f"route_prefix={self.route_prefix})"
        )

    def __repr__(self):
        return str(self)


def deployment_to_schema(
    d: Deployment, include_route_prefix: bool = True
) -> DeploymentSchema:
    """Converts a live deployment object to a corresponding structured schema.

    Args:
        d: Deployment object to convert
        include_route_prefix: Whether to include the route_prefix in the returned
            schema. This should be set to False if the schema will be included in a
            higher-level object describing an application, and you want to place
            route_prefix at the application level.
    """

    if d.ray_actor_options is not None:
        ray_actor_options_schema = RayActorOptionsSchema.parse_obj(d.ray_actor_options)
    else:
        ray_actor_options_schema = None

    deployment_options = {
        "name": d.name,
        "num_replicas": None
        if d._deployment_config.autoscaling_config
        else d.num_replicas,
        "max_ongoing_requests": d.max_ongoing_requests,
        "max_queued_requests": d.max_queued_requests,
        "user_config": d.user_config,
        "autoscaling_config": d._deployment_config.autoscaling_config,
        "graceful_shutdown_wait_loop_s": d._deployment_config.graceful_shutdown_wait_loop_s,  # noqa: E501
        "graceful_shutdown_timeout_s": d._deployment_config.graceful_shutdown_timeout_s,
        "health_check_period_s": d._deployment_config.health_check_period_s,
        "health_check_timeout_s": d._deployment_config.health_check_timeout_s,
        "ray_actor_options": ray_actor_options_schema,
        "placement_group_strategy": d._replica_config.placement_group_strategy,
        "placement_group_bundles": d._replica_config.placement_group_bundles,
        "max_replicas_per_node": d._replica_config.max_replicas_per_node,
        "logging_config": d._deployment_config.logging_config,
    }

    if include_route_prefix:
        deployment_options["route_prefix"] = d.route_prefix

    # Let non-user-configured options be set to defaults. If the schema
    # is converted back to a deployment, this lets Serve continue tracking
    # which options were set by the user. Name is a required field in the
    # schema, so it should be passed in explicitly.
    for option in list(deployment_options.keys()):
        if (
            option != "name"
            and option not in d._deployment_config.user_configured_option_names
        ):
            del deployment_options[option]

    # TODO(Sihan) DeploymentConfig num_replicas and auto_config can be set together
    # because internally we use these two field for autoscale and deploy.
    # We can improve the code after we separate the user faced deployment config and
    # internal deployment config.
    return DeploymentSchema(**deployment_options)


def schema_to_deployment(s: DeploymentSchema) -> Deployment:
    """Creates a deployment with parameters specified in schema.

    The returned deployment CANNOT be deployed immediately. It's func_or_class
    value is an empty string (""), which is not a valid import path. The
    func_or_class value must be overwritten with a valid function or class
    before the deployment can be deployed.
    """

    if s.ray_actor_options is DEFAULT.VALUE:
        ray_actor_options = None
    else:
        ray_actor_options = s.ray_actor_options.dict(exclude_unset=True)

    if s.placement_group_bundles is DEFAULT.VALUE:
        placement_group_bundles = None
    else:
        placement_group_bundles = s.placement_group_bundles

    if s.placement_group_strategy is DEFAULT.VALUE:
        placement_group_strategy = None
    else:
        placement_group_strategy = s.placement_group_strategy

    if s.max_replicas_per_node is DEFAULT.VALUE:
        max_replicas_per_node = None
    else:
        max_replicas_per_node = s.max_replicas_per_node

    deployment_config = DeploymentConfig.from_default(
        num_replicas=s.num_replicas,
        user_config=s.user_config,
        max_ongoing_requests=s.max_ongoing_requests,
        max_queued_requests=s.max_queued_requests,
        autoscaling_config=s.autoscaling_config,
        graceful_shutdown_wait_loop_s=s.graceful_shutdown_wait_loop_s,
        graceful_shutdown_timeout_s=s.graceful_shutdown_timeout_s,
        health_check_period_s=s.health_check_period_s,
        health_check_timeout_s=s.health_check_timeout_s,
        logging_config=s.logging_config,
    )
    deployment_config.user_configured_option_names = (
        s._get_user_configured_option_names()
    )

    replica_config = ReplicaConfig.create(
        deployment_def="",
        init_args=(),
        init_kwargs={},
        ray_actor_options=ray_actor_options,
        placement_group_bundles=placement_group_bundles,
        placement_group_strategy=placement_group_strategy,
        max_replicas_per_node=max_replicas_per_node,
    )

    return Deployment(
        name=s.name,
        deployment_config=deployment_config,
        replica_config=replica_config,
        route_prefix=s.route_prefix,
        _internal=True,
    )
