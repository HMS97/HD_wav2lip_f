import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import face_detection
import skvideo.io
from path import Path
parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=48, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):

	vidname = os.path.basename(vfile).split('/')[-1]
	dirname = 'HQ_face'

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	# print(fulldir)
	os.makedirs(fulldir, exist_ok=True)
	frames_name = sorted(Path(vfile).files(), key = lambda x: int(x.split("/")[-1].split(".")[0]))

	batches = [frames_name[i:i + args.batch_size] for i in range(0, len(frames_name), args.batch_size)]


	i = -1
	for fb_names in batches:
		fb = []
		ori_list = []
		lxy_list = []
		for item in fb_names:

			temp = cv2.imread(item)
			ori_list.append(temp)
			lx = temp.shape[1]/hp.img_size
			ly = temp.shape[0]/hp.img_size
			temp = cv2.resize(temp,(hp.img_size,hp.img_size))
			fb.append(temp)
			lxy_list.append((lx,ly))

		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue
			x1, y1, x2, y2 = f
			# print(lxy_list[j][1], lxy_list[j][0])
			# print(int(lxy_list[j][1]*y1),int(lxy_list[j][1]*y2), int(lxy_list[j][0]*x1),int(lxy_list[j][0]*x2), ori_list[j].shape)
			cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), ori_list[j][int(lxy_list[j][1]*y1):int(lxy_list[j][1]*y2), int(lxy_list[j][0]*x1):int(lxy_list[j][0]*x2)])


def split_video_file(vfile, args, gpu_id):
	video_stream = cv2.VideoCapture(vfile)
	# video_stream = skvideo.io.VideoCapture(vfile)
	frames = []
	ori_frames = []
	i = -1

	vidname = os.path.basename(vfile).split('.')[0]
	dirname = 'HQ_Images'
	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	# print(fulldir)
	while 1:
		still_reading, frame = video_stream.read()

		if not still_reading:
			video_stream.release()
			break


		i = i+1
		cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), frame)

		
		

def process_audio_file(vfile, args):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = 'HQ_face'

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')
	command = template.format(vfile, wavpath)
	subprocess.call(command, shell=True)

	
def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		# process_video_file(vfile, args, gpu_id)
		split_video_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()


def mp_handler_pd(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))
	if os.path.exists(args.preprocessed_root):
		pass
	else:
		os.mkdir(f'{args.preprocessed_root}')
		os.mkdir(f'{args.preprocessed_root}/HQ_face')
		os.mkdir(f'{args.preprocessed_root}/HQ_Images')



	# filelist = ['main/videos/video-0-0-1a.mp4','main/videos/video-1-0-4a.mp4','main/videos/video-2-0-4a.mp4','main/videos/video-3-0-4a.mp4']
	# filelist = glob(path.join(args.data_root, '*/*.mp4'))
	# # print(filelist)
	# filelist = [i for i in filelist if len(i)>2]
	# filelist = sorted(filelist)
	# # print(filelist)
	# jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	# p = ThreadPoolExecutor(args.ngpu)
	# futures = [p.submit(mp_handler, j) for j in jobs]
	# _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	# filelist = glob(path.join(args.data_root, '*\*.mp4'))
	filelist = glob(path.join(args.data_root, '*.mp4'))
	folder_list = Path(f'{args.preprocessed_root}/HQ_Images').listdir()
	print(folder_list)
	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(folder_list)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler_pd, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]



	# print('Dumping audios...')

	# for vfile in tqdm(filelist):
	# 	try:
	# 		process_audio_file(vfile, args)
	# 	except KeyboardInterrupt:
	# 		exit(0)
	# 	except:
	# 		traceback.print_exc()
	# 		continue

if __name__ == '__main__':
	main(args)


# cmake -D CMAKE_BUILD_TYPE=Release -D WITH_FFMPEG=ON -D CMAKE_INSTALL_PREFIX=/usr/local PYTHON3_EXECUTABLE = /home/huimingsun/anaconda3/bin/python3.7 PYTHON_INCLUDE_DIR = /usr/include/python3.5m  ..
