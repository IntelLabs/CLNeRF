gdown --id 1iAE8iWtUokQ6waZYc9ysy2OFkXyptdso -O ../../dataset/WOT.zip
unzip ../../dataset/WOT.zip -d ../../dataset
rm ../../dataset/WOT.zip

# if you want to rerun the colmap reconstruction, please uncomment (and change the name to your customized dataset folder) this code if you want to prepare WOT-type dataset from a sequence of videos
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/breville
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/community
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/kitchen
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/living_room
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/spa
# python ../../utils/data_prepare_utils/poses/imgs2poses.py --is_video 1 --frame_rate 20 ../../dataset/WOT/street
