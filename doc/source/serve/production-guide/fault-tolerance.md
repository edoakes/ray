(serve-e2e-ft)=
# Add End-to-End Fault Tolerance

This section helps you:

* Provide additional fault tolerance for your Serve application
* Understand Serve's recovery procedures
* Simulate system errors in your Serve application

:::{admonition} Relevant Guides
:class: seealso
This section discusses concepts from:
* Serve's [architecture guide](serve-architecture)
* Serve's [Kubernetes production guide](serve-in-production-kubernetes)
:::

(serve-e2e-ft-guide)=
## Guide: end-to-end fault tolerance for your Serve app

Serve provides some [fault tolerance](serve-ft-detail) features out of the box. Two options to get end-to-end fault tolerance are the following:
* tune these features and run Serve on top of [KubeRay]
* use the [Anyscale platform](https://docs.anyscale.com/platform/services/head-node-ft?utm_source=ray_docs&utm_medium=docs&utm_campaign=tolerance), a managed Ray platform

### Replica health-checking

By default, the Serve controller periodically health-checks each Serve deployment replica and restarts it on failure.

You can define custom application-level health-checks and adjust their frequency and timeout.
To define a custom health-check, add a `check_health` method to your deployment class.
This method should take no arguments and return no result, and it should raise an exception if Ray Serve considers the replica unhealthy.
If the health-check fails, the Serve controller logs the exception, kills the unhealthy replica(s), and restarts them.
You can also use the deployment options to customize how frequently Serve runs the health-check and the timeout after which Serve marks a replica unhealthy.

```{literalinclude} ../doc_code/fault_tolerance/replica_health_check.py
:start-after: __health_check_start__
:end-before: __health_check_end__
:language: python
```

In this example, `check_health` raises an error if the connection to an external database is lost. The Serve controller periodically calls this method on each replica of the deployment. If the method raises an exception for a replica, Serve marks that replica as unhealthy and restarts it. Health checks are configured and performed on a per-replica basis.

:::{note}
You shouldn't call ``check_health`` directly through a deployment handle (e.g., ``await deployment_handle.check_health.remote()``). This would invoke the health check on a single, arbitrary replica. The ``check_health`` method is designed as an interface for the Serve controller, not for direct user calls.
:::

:::{note}
In a composable deployment graph, each deployment is responsible for its own health, independent of the other deployments it's bound to. For example, in an application defined by ``app = ParentDeployment.bind(ChildDeployment.bind())``, ``ParentDeployment`` doesn't restart if ``ChildDeployment`` replicas fail their health checks. When the ``ChildDeployment`` replicas recover, the handle in ``ParentDeployment`` updates automatically to route requests to the healthy replicas.
:::

### Worker node recovery

:::{admonition} KubeRay Required
:class: caution, dropdown
You **must** deploy your Serve application with [KubeRay] to use this feature.

See Serve's [Kubernetes production guide](serve-in-production-kubernetes) to learn how you can deploy your app with KubeRay.
:::

By default, Serve can recover from certain failures, such as unhealthy actors. When [Serve runs on Kubernetes](serve-in-production-kubernetes) with [KubeRay], it can also recover from some cluster-level failures, such as dead workers or head nodes.

When a worker node fails, the actors running on it also fail. Serve detects that the actors have failed, and it attempts to respawn the actors on the remaining, healthy nodes. Meanwhile, KubeRay detects that the node itself has failed, so it attempts to restart the worker pod on another running node, and it also brings up a new healthy node to replace it. Once the node comes up, if the pod is still pending, it can be restarted on that node. Similarly, Serve can also respawn any pending actors on that node as well. The deployment replicas running on healthy nodes can continue serving traffic throughout the recovery period.

(serve-e2e-ft-guide-gcs)=
### Head node recovery: Ray GCS fault tolerance

:::{admonition} KubeRay Required
:class: caution, dropdown
You **must** deploy your Serve application with [KubeRay] to use this feature.

See Serve's [Kubernetes production guide](serve-in-production-kubernetes) to learn how you can deploy your app with KubeRay.
:::

In this section, you'll learn how to add fault tolerance to Ray's Global Control Store (GCS), which allows your Serve application to serve traffic even when the head node crashes.

By default, the Ray head node is a single point of failure: if it crashes, the entire Ray cluster crashes and you must restart it. When running on Kubernetes, the `RayService` controller health-checks the Ray cluster and restarts it if this occurs, but this introduces some downtime.

Starting with Ray 2.0+, KubeRay supports [Global Control Store (GCS) fault tolerance](kuberay-gcs-ft), preventing the Ray cluster from crashing if the head node goes down.
While the head node is recovering, Serve applications can still handle traffic with worker nodes but you can't update or recover from other failures like Actors or Worker nodes crashing.
Once the GCS recovers, the cluster returns to normal behavior.

You can enable GCS fault tolerance on KubeRay by adding an external Redis server and modifying your `RayService` Kubernetes object with the following steps:

#### Step 1: Add external Redis server

GCS fault tolerance requires an external Redis database. You can choose to host your own Redis database, or you can use one through a third-party vendor. Use a highly available Redis database for resiliency.

**For development purposes**, you can also host a small Redis database on the same Kubernetes cluster as your Ray cluster. For example, you can add a 1-node Redis cluster by prepending these three Redis objects to your Kubernetes YAML:

(one-node-redis-example)=
```YAML
kind: ConfigMap
apiVersion: v1
metadata:
  name: redis-config
  labels:
    app: redis
data:
  redis.conf: |-
    port 6379
    bind 0.0.0.0
    protected-mode no
    requirepass 5241590000000000
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  labels:
    app: redis
spec:
  type: ClusterIP
  ports:
    - name: redis
      port: 6379
  selector:
    app: redis
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:5.0.8
          command:
            - "sh"
            - "-c"
            - "redis-server /usr/local/etc/redis/redis.conf"
          ports:
            - containerPort: 6379
          volumeMounts:
            - name: config
              mountPath: /usr/local/etc/redis/redis.conf
              subPath: redis.conf
      volumes:
        - name: config
          configMap:
            name: redis-config
---
```

**This configuration is NOT production-ready**, but it's useful for development and testing. When you move to production, it's highly recommended that you replace this 1-node Redis cluster with a highly available Redis cluster.

#### Step 2: Add Redis info to RayService

After adding the Redis objects, you also need to modify the `RayService` configuration.

First, you need to update your `RayService` metadata's annotations:

::::{tab-set}

:::{tab-item} Vanilla Config
```yaml
...
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: rayservice-sample
spec:
...
```
:::

:::{tab-item} Fault Tolerant Config
:selected:
```yaml
...
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: rayservice-sample
  annotations:
    ray.io/ft-enabled: "true"
    ray.io/external-storage-namespace: "my-raycluster-storage-namespace"
spec:
...
```
:::

::::

The annotations are:
* `ray.io/ft-enabled` REQUIRED: Enables GCS fault tolerance when true
* `ray.io/external-storage-namespace` OPTIONAL: Sets the [external storage namespace]

Next, you need to add the `RAY_REDIS_ADDRESS` environment variable to the `headGroupSpec`:

::::{tab-set}

:::{tab-item} Vanilla Config

```yaml
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
    ...
spec:
    ...
    rayClusterConfig:
        headGroupSpec:
            ...
            template:
                ...
                spec:
                    ...
                    env:
                        ...
```

:::

:::{tab-item} Fault Tolerant Config
:selected:

```yaml
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
    ...
spec:
    ...
    rayClusterConfig:
        headGroupSpec:
            ...
            template:
                ...
                spec:
                    ...
                    env:
                        ...
                        - name: RAY_REDIS_ADDRESS
                          value: redis:6379
```
:::

::::

`RAY_REDIS_ADDRESS`'s value should be your Redis database's `redis://` address. It should contain your Redis database's host and port. An [example Redis address](https://www.iana.org/assignments/uri-schemes/prov/rediss) is `redis://user:secret@localhost:6379/0?foo=bar&qux=baz`.

In the example above, the Redis deployment name (`redis`) is the host within the Kubernetes cluster, and the Redis port is `6379`. The example is compatible with the previous section's [example config](one-node-redis-example).

After you apply the Redis objects along with your updated `RayService`, your Ray cluster can recover from head node crashes without restarting all the workers!

:::{seealso}
Check out the KubeRay guide on [GCS fault tolerance](kuberay-gcs-ft) to learn more about how Serve leverages the external Redis cluster to provide head node fault tolerance.
:::

### Spreading replicas across nodes

One way to improve the availability of your Serve application is to spread deployment replicas across multiple nodes so that you still have enough running
replicas to serve traffic even after a certain number of node failures.

By default, Serve soft spreads all deployment replicas but it has a few limitations:

* The spread is soft and best-effort with no guarantee that the it's perfectly even.

* Serve tries to spread replicas among the existing nodes if possible instead of launching new nodes.
For example, if you have a big enough single node cluster, Serve schedules all replicas on that single node assuming
it has enough resources. However, that node becomes the single point of failure.

You can change the spread behavior of your deployment with the `max_replicas_per_node`
[deployment option](../../serve/api/doc/ray.serve.deployment_decorator.rst), which hard limits the number of replicas of a given deployment that can run on a single node.
If you set it to 1 then you're effectively strict spreading the deployment replicas. If you don't set it then there's no hard spread constraint and Serve uses the default soft spread mentioned in the preceding paragraph. `max_replicas_per_node` option is per deployment and only affects the spread of replicas within a deployment. There's no spread between replicas of different deployments.

The following code example shows how to set `max_replicas_per_node` deployment option:

```{testcode}
import ray
from ray import serve

@serve.deployment(max_replicas_per_node=1)
class Deployment1:
  def __call__(self, request):
    return "hello"

@serve.deployment(max_replicas_per_node=2)
class Deployment2:
  def __call__(self, request):
    return "world"
```

This example has two Serve deployments with different `max_replicas_per_node`: `Deployment1` can have at most one replica on each node and `Deployment2` can have at most two replicas on each node. If you schedule two replicas of `Deployment1` and two replicas of `Deployment2`, Serve runs a cluster with at least two nodes, each running one replica of `Deployment1`. The two replicas of `Deployment2` may run on either a single node or across two nodes because either satisfies the `max_replicas_per_node` constraint.

(serve-e2e-ft-behavior)=
## Serve's recovery procedures

This section explains how Serve recovers from system failures. It uses the following Serve application and config as a working example.

::::{tab-set}

:::{tab-item} Python Code
```{literalinclude} ../doc_code/fault_tolerance/sleepy_pid.py
:start-after: __start__
:end-before: __end__
:language: python
```
:::

:::{tab-item} Kubernetes Config
```{literalinclude} ../doc_code/fault_tolerance/k8s_config.yaml
:language: yaml
```
:::

::::

Follow the [KubeRay quickstart guide](kuberay-quickstart) to:
* Install `kubectl` and `Helm`
* Prepare a Kubernetes cluster
* Deploy a KubeRay operator

Then, [deploy the Serve application](serve-deploy-app-on-kuberay) above:

```console
$ kubectl apply -f config.yaml
```

### Worker node failure

You can simulate a worker node failure in the working example. First, take a look at the nodes and pods running in your Kubernetes cluster:

```console
$ kubectl get nodes

NAME                                        STATUS   ROLES    AGE     VERSION
gke-serve-demo-default-pool-ed597cce-nvm2   Ready    <none>   3d22h   v1.22.12-gke.1200
gke-serve-demo-default-pool-ed597cce-m888   Ready    <none>   3d22h   v1.22.12-gke.1200
gke-serve-demo-default-pool-ed597cce-pu2q   Ready    <none>   3d22h   v1.22.12-gke.1200

$ kubectl get pods -o wide

NAME                                                      READY   STATUS    RESTARTS        AGE    IP           NODE                                        NOMINATED NODE   READINESS GATES
ervice-sample-raycluster-thwmr-worker-small-group-bdv6q   1/1     Running   0               3m3s   10.68.2.62   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
ervice-sample-raycluster-thwmr-worker-small-group-pztzk   1/1     Running   0               3m3s   10.68.2.61   gke-serve-demo-default-pool-ed597cce-m888   <none>           <none>
rayservice-sample-raycluster-thwmr-head-28mdh             1/1     Running   1 (2m55s ago)   3m3s   10.68.0.45   gke-serve-demo-default-pool-ed597cce-pu2q   <none>           <none>
redis-75c8b8b65d-4qgfz                                    1/1     Running   0               3m3s   10.68.2.60   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
```

Open a separate terminal window and port-forward to one of the worker nodes:

```console
$ kubectl port-forward ervice-sample-raycluster-thwmr-worker-small-group-bdv6q 8000
Forwarding from 127.0.0.1:8000 -> 8000
Forwarding from [::1]:8000 -> 8000
```

While the `port-forward` is running, you can query the application in another terminal window:

```console
$ curl localhost:8000
418
```

The output is the process ID of the deployment replica that handled the request. The application launches 6 deployment replicas, so if you run the query multiple times, you should see different process IDs:

```console
$ curl localhost:8000
418
$ curl localhost:8000
256
$ curl localhost:8000
385
```

Now you can simulate worker failures. You have two options: kill a worker pod or kill a worker node. Let's start with the worker pod. Make sure to kill the pod that you're **not** port-forwarding to, so you can continue querying the living worker while the other one relaunches.

```console
$ kubectl delete pod ervice-sample-raycluster-thwmr-worker-small-group-pztzk
pod "ervice-sample-raycluster-thwmr-worker-small-group-pztzk" deleted

$ curl localhost:8000
6318
```

While the pod crashes and recovers, the live pod can continue serving traffic!

:::{tip}
Killing a node and waiting for it to recover usually takes longer than killing a pod and waiting for it to recover. For this type of debugging, it's quicker to simulate failures by killing at the pod level rather than at the node level.
:::

You can similarly kill a worker node and see that the other nodes can continue serving traffic:

```console
$ kubectl get pods -o wide

NAME                                                      READY   STATUS    RESTARTS      AGE     IP           NODE                                        NOMINATED NODE   READINESS GATES
ervice-sample-raycluster-thwmr-worker-small-group-bdv6q   1/1     Running   0             65m     10.68.2.62   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
ervice-sample-raycluster-thwmr-worker-small-group-mznwq   1/1     Running   0             5m46s   10.68.1.3    gke-serve-demo-default-pool-ed597cce-m888   <none>           <none>
rayservice-sample-raycluster-thwmr-head-28mdh             1/1     Running   1 (65m ago)   65m     10.68.0.45   gke-serve-demo-default-pool-ed597cce-pu2q   <none>           <none>
redis-75c8b8b65d-4qgfz                                    1/1     Running   0             65m     10.68.2.60   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>

$ kubectl delete node gke-serve-demo-default-pool-ed597cce-m888
node "gke-serve-demo-default-pool-ed597cce-m888" deleted

$ curl localhost:8000
385
```

### Head node failure

You can simulate a head node failure by either killing the head pod or the head node. First, take a look at the running pods in your cluster:

```console
$ kubectl get pods -o wide

NAME                                                      READY   STATUS    RESTARTS      AGE     IP           NODE                                        NOMINATED NODE   READINESS GATES
ervice-sample-raycluster-thwmr-worker-small-group-6f2pk   1/1     Running   0             6m59s   10.68.2.64   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
ervice-sample-raycluster-thwmr-worker-small-group-bdv6q   1/1     Running   0             79m     10.68.2.62   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
rayservice-sample-raycluster-thwmr-head-28mdh             1/1     Running   1 (79m ago)   79m     10.68.0.45   gke-serve-demo-default-pool-ed597cce-pu2q   <none>           <none>
redis-75c8b8b65d-4qgfz                                    1/1     Running   0             79m     10.68.2.60   gke-serve-demo-default-pool-ed597cce-nvm2   <none>           <none>
```

Port-forward to one of your worker pods. Make sure this pod is on a separate node from the head node, so you can kill the head node without crashing the worker:

```console
$ kubectl port-forward ervice-sample-raycluster-thwmr-worker-small-group-bdv6q
Forwarding from 127.0.0.1:8000 -> 8000
Forwarding from [::1]:8000 -> 8000
```

In a separate terminal, you can make requests to the Serve application:

```console
$ curl localhost:8000
418
```

You can kill the head pod to simulate killing the Ray head node:

```console
$ kubectl delete pod rayservice-sample-raycluster-thwmr-head-28mdh
pod "rayservice-sample-raycluster-thwmr-head-28mdh" deleted

$ curl localhost:8000
```

If you have configured [GCS fault tolerance](serve-e2e-ft-guide-gcs) on your cluster, your worker pod can continue serving traffic without restarting when the head pod crashes and recovers. Without GCS fault tolerance, KubeRay restarts all worker pods when the head pod crashes, so you'll need to wait for the workers to restart and the deployments to reinitialize before you can port-forward and send more requests.

### Serve controller failure

You can simulate a Serve controller failure by manually killing the Serve actor.

If you're running KubeRay, `exec` into one of your pods:

```console
$ kubectl get pods

NAME                                                      READY   STATUS    RESTARTS   AGE
ervice-sample-raycluster-mx5x6-worker-small-group-hfhnw   1/1     Running   0          118m
ervice-sample-raycluster-mx5x6-worker-small-group-nwcpb   1/1     Running   0          118m
rayservice-sample-raycluster-mx5x6-head-bqjhw             1/1     Running   0          118m
redis-75c8b8b65d-4qgfz                                    1/1     Running   0          3h36m

$ kubectl exec -it rayservice-sample-raycluster-mx5x6-head-bqjhw -- bash
ray@rayservice-sample-raycluster-mx5x6-head-bqjhw:~$
```

You can use the [Ray State API](state-api-cli-ref) to inspect your Serve app:

```console
$ ray summary actors

======== Actors Summary: 2022-10-04 21:06:33.678706 ========
Stats:
------------------------------------
total_actors: 10


Table (group by class):
------------------------------------
    CLASS_NAME              STATE_COUNTS
0   ProxyActor          ALIVE: 3
1   ServeReplica:SleepyPid  ALIVE: 6
2   ServeController         ALIVE: 1

$ ray list actors --filter "class_name=ServeController"

======== List: 2022-10-04 21:09:14.915881 ========
Stats:
------------------------------
Total: 1

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME       STATE    NAME                      PID
 0  70a718c973c2ce9471d318f701000000  ServeController  ALIVE    SERVE_CONTROLLER_ACTOR  48570
```

You can then kill the Serve controller via the Python interpreter. Note that you'll need to use the `NAME` from the `ray list actor` output to get a handle to the Serve controller.

```console
$ python

>>> import ray
>>> controller_handle = ray.get_actor("SERVE_CONTROLLER_ACTOR", namespace="serve")
>>> ray.kill(controller_handle, no_restart=True)
>>> exit()
```

You can use the Ray State API to check the controller's status:

```console
$ ray list actors --filter "class_name=ServeController"

======== List: 2022-10-04 21:36:37.157754 ========
Stats:
------------------------------
Total: 2

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME       STATE    NAME                      PID
 0  3281133ee86534e3b707190b01000000  ServeController  ALIVE    SERVE_CONTROLLER_ACTOR  49914
 1  70a718c973c2ce9471d318f701000000  ServeController  DEAD     SERVE_CONTROLLER_ACTOR  48570
```

You should still be able to query your deployments while the controller is recovering:

```
# If you're running KubeRay, you
# can do this from inside the pod:

$ python

>>> import requests
>>> requests.get("http://localhost:8000").json()
347
```

:::{note}
While the controller is dead, replica health-checking and deployment autoscaling will not work. They'll continue working once the controller recovers.
:::

### Deployment replica failure

You can simulate replica failures by manually killing deployment replicas. If you're running KubeRay, make sure to `exec` into a Ray pod before running these commands.

```console
$ ray summary actors

======== Actors Summary: 2022-10-04 21:40:36.454488 ========
Stats:
------------------------------------
total_actors: 11


Table (group by class):
------------------------------------
    CLASS_NAME              STATE_COUNTS
0   ProxyActor          ALIVE: 3
1   ServeController         ALIVE: 1
2   ServeReplica:SleepyPid  ALIVE: 6

$ ray list actors --filter "class_name=ServeReplica:SleepyPid"

======== List: 2022-10-04 21:41:32.151864 ========
Stats:
------------------------------
Total: 6

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME              STATE    NAME                               PID
 0  39e08b172e66a5d22b2b4cf401000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#RlRptP    203
 1  55d59bcb791a1f9353cd34e301000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#BnoOtj    348
 2  8c34e675edf7b6695461d13501000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#SakmRM    283
 3  a95405318047c5528b7483e701000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#rUigUh    347
 4  c531188fede3ebfc868b73a001000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#gbpoFe    383
 5  de8dfa16839443f940fe725f01000000  ServeReplica:SleepyPid  ALIVE    SERVE_REPLICA::SleepyPid#PHvdJW    176
```

You can use the `NAME` from the `ray list actor` output to get a handle to one of the replicas:

```console
$ python

>>> import ray
>>> replica_handle = ray.get_actor("SERVE_REPLICA::SleepyPid#RlRptP", namespace="serve")
>>> ray.kill(replica_handle, no_restart=True)
>>> exit()
```

While the replica is restarted, the other replicas can continue processing requests. Eventually the replica restarts and continues serving requests:

```console
$ python

>>> import requests
>>> requests.get("http://localhost:8000").json()
383
```

### Proxy failure

You can simulate Proxy failures by manually killing `ProxyActor` actors. If you're running KubeRay, make sure to `exec` into a Ray pod before running these commands.

```console
$ ray summary actors

======== Actors Summary: 2022-10-04 21:51:55.903800 ========
Stats:
------------------------------------
total_actors: 12


Table (group by class):
------------------------------------
    CLASS_NAME              STATE_COUNTS
0   ProxyActor          ALIVE: 3
1   ServeController         ALIVE: 1
2   ServeReplica:SleepyPid  ALIVE: 6

$ ray list actors --filter "class_name=ProxyActor"

======== List: 2022-10-04 21:52:39.853758 ========
Stats:
------------------------------
Total: 3

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME      STATE    NAME                                                                                                 PID
 0  283fc11beebb6149deb608eb01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-91f9a685e662313a0075efcb7fd894249a5bdae7ee88837bea7985a0    101
 1  2b010ce28baeff5cb6cb161e01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-cc262f3dba544a49ea617d5611789b5613f8fe8c86018ef23c0131eb    133
 2  7abce9dd241b089c1172e9ca01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-7589773fc62e08c2679847aee9416805bbbf260bee25331fa3389c4f    267
```

You can use the `NAME` from the `ray list actor` output to get a handle to one of the replicas:

```console
$ python

>>> import ray
>>> proxy_handle = ray.get_actor("SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-91f9a685e662313a0075efcb7fd894249a5bdae7ee88837bea7985a0", namespace="serve")
>>> ray.kill(proxy_handle, no_restart=False)
>>> exit()
```

While the proxy is restarted, the other proxies can continue accepting requests. Eventually the proxy restarts and continues accepting requests. You can use the `ray list actor` command to see when the proxy restarts:

```console
$ ray list actors --filter "class_name=ProxyActor"

======== List: 2022-10-04 21:58:41.193966 ========
Stats:
------------------------------
Total: 3

Table:
------------------------------
    ACTOR_ID                          CLASS_NAME      STATE    NAME                                                                                                 PID
 0  283fc11beebb6149deb608eb01000000  ProxyActor  ALIVE     SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-91f9a685e662313a0075efcb7fd894249a5bdae7ee88837bea7985a0  57317
 1  2b010ce28baeff5cb6cb161e01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-cc262f3dba544a49ea617d5611789b5613f8fe8c86018ef23c0131eb    133
 2  7abce9dd241b089c1172e9ca01000000  ProxyActor  ALIVE    SERVE_CONTROLLER_ACTOR:SERVE_PROXY_ACTOR-7589773fc62e08c2679847aee9416805bbbf260bee25331fa3389c4f    267
```

Note that the PID for the first ProxyActor has changed, indicating that it restarted.

[KubeRay]: kuberay-index
[external storage namespace]: kuberay-external-storage-namespace
