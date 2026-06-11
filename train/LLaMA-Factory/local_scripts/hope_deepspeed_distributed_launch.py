import json
import os

if "AFO_ENV_CLUSTER_SPEC" in os.environ:
    cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    afo_spec = json.loads(os.environ["AFO_SPEC"])
    role = cluster_spec["role"]
    assert role == "worker", "{} vs worker".format(role)
    node_rank = cluster_spec["index"]
    nnodes = len(cluster_spec[role])
    nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
    master = afo_spec['cluster'][role][0]
    master_addr, master_ports = master.split(":")
    master_ports = master_ports.split(",")
    master_port = master_ports[0]
    print(f"export FORCE_TORCHRUN=1")
    print(f"export NNODES={nnodes}")
    print(f"export NODE_RANK={node_rank}")
    print(f"export MASTER_ADDR={master_addr}")
    print(f"export MASTER_PORT={master_port}")
else:
   # in case of hope workbench debugging
#    nproc_per_node = 2
#    nnodes = 1
#    node_rank = 0
#    master_addr = "localhost"
#    master_port = "3333"
#    print(
#        "torchrun "
#        "--nproc_per_node={} "
#        "--nnodes={} "
#        "--node_rank={} "
#        "--master_addr={} "
#        "--master_port={}".format(
#            nproc_per_node, nnodes, node_rank, master_addr, master_port
#        )
#    )
    print(f"export FORCE_TORCHRUN=1")
    print(f"export NNODES=1")
    print(f"export NODE_RANK=0")
    print(f"export MASTER_ADDR=localhost")
    print(f"export MASTER_PORT=29501")