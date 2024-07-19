## semantic kitti
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63430 --use_env main.py config_server_kitti.yaml
## nuscenes
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=61425 --use_env main.py config_server_nus.yaml

# a2d2
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=61425 --use_env main.py config_server_a2d2.yaml
