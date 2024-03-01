import cv2
import numpy as np
import sys, os
import joblib
from tkinter import filedialog
from tqdm import tqdm

import subprocess
from icecream import ic

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
	try:
		n = np.fromfile(filename, dtype)
		img = cv2.imdecode(n, flags)
		return img
	except Exception as e:
		print(e)
		return None

def imwrite(filename, img, params=None):
	try:
		ext = os.path.splitext(filename)[1]
		result, n = cv2.imencode(ext, img, params)

		if result:
			with open(filename, mode='w+b') as f:
				n.tofile(f)
			return True
		else:
			return False
	except Exception as e:
		print(e)
		return False

def get_file_list(dir:str):
	file_or_dir_list = os.listdir(dir)
	file_list = [os.path.join(dir, f) for f in file_or_dir_list if os.path.isfile(os.path.join(dir, f))]
	return file_list

def resource_path(relative_path):
	if hasattr(sys, '_MEIPASS'):
		return os.path.join(sys._MEIPASS, relative_path)
	return os.path.join(os.path.abspath("."), relative_path)

"""
def get_dir_list(dir:str):
	file_or_dir_list = os.listdir(dir)
	dir_list = [os.path.join(dir, f) for f in file_or_dir_list if os.path.isdir(os.path.join(dir, f))]
	return dir_list
"""

def floatrange(start, end, step):
	result = []
	i = start
	while i <= end:
		result.append(i)
		i += step
	if result[-1] != end:
		result.append(end)
	return result

def unpack_rgba(img:np.ndarray):
	h, w, c = img.shape
	if c == 4:
		return img[:, :, :3], img[:, :, 3] > 0
	else:
		return img, np.ones((h, w), dtype=bool)

"""
# 平均二乗誤差
def img_mse(img1:np.ndarray, img2:np.ndarray):
	# サイズを揃える
	h1, w1, _ = img1.shape
	h2, w2, _ = img2.shape
	if w1 < w2:
		img2 = cv2.resize(img2, (w1, w1))
	elif w1 > w2:
		img1 = cv2.resize(img1, (w2, w2))

	img1, mask1 = unpack_rgba(img1)
	img2, mask2 = unpack_rgba(img2)
	mask = mask1 & mask2

	error = img2[mask] - img1[mask]
	squared_error = error**2
	mean_squared_error = np.mean(squared_error)

	return mean_squared_error
"""

def scaledMatchTemplate(image:np.ndarray, template:np.ndarray, scale=1, **kwargs):
	size = int(min(image.shape[:2]) * scale)
	template = cv2.resize(template, (size, size))
	if 'mask' in kwargs:
		kwargs['mask'] = cv2.resize(kwargs['mask'], (size, size))
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, **kwargs))
	# return [scale, minVal, *minLoc]
	return [scale, maxVal, *maxLoc]

def scaledMatchTemplateSearch(image:np.ndarray, template:np.ndarray, scales:list, **kwargs):
	result = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(scaledMatchTemplate)(image, template, scale, **kwargs) for scale in tqdm(scales))
	result = np.array(result)
	best_result = result[np.argmax(result, axis=0)[1]]
	return best_result

def twoStepScaledMatchTemplateSearch(image:np.ndarray, template:np.ndarray, min_scale, max_scale, **kwargs):
	step = 1 / min(image.shape[:2])
	scale, score, x1, y1 = scaledMatchTemplateSearch(image, template, floatrange(min_scale, max_scale, step*2))
	return scaledMatchTemplateSearch(image, template, floatrange(max(scale-0.02, min_scale), min(scale+0.02, max_scale), step))


if __name__ == '__main__':
	# multiprocessing.freeze_support()
	if len(sys.argv) > 2:
		input_files = sys.argv[1:3]
	else:
		filetypes = [('画像', '*.png;*.jpg;*.jpeg')]
		input_files = filedialog.askopenfilenames(initialdir=os.getcwd(), filetypes=filetypes)

	if len(input_files) != 2:
		print('エラー: 入力ファイルは2つにしてください')
		exit(-1)

	print('input files:')
	print(input_files[0])
	print(input_files[1])
	input_imgs = [imread(f) for f in input_files]

	# 入力のうちどちらがフォトでどちらがアイコンか判定
	h, w, _ = input_imgs[0].shape
	r = h / w
	if r > 1.5 or r < 0.666:
		photo = input_imgs[0]
		icon = input_imgs[1]
	else:
		photo = input_imgs[1]
		icon = input_imgs[0]

	############## フォトの前処理 ##############
	# フォトが1024x720より大きければ縮小
	h, w, _ = photo.shape
	r = 1024 / max(h, w)
	if r < 1:
		photo = cv2.resize(photo, None, fx=r, fy=r)
		h, w, _ = photo.shape

	# フォトが横長なら90°回転
	if h < w:
		photo = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
		h, w, _ = photo.shape

	# 余白をつける
	photo_ = np.zeros_like(photo)
	photo_ = cv2.resize(photo_, None, fx=1.1, fy=1.1)
	h_, w_, _ = photo_.shape
	x, y = (w_ - w) // 2, (h_ - h) // 2
	photo_[y:y+h, x:x+w, :] = photo
	photo = photo_

	############## アイコンの前処理 ##############
	# アイコンが360x360より大きければ縮小
	r = 360 / max(icon.shape[:2])
	if r < 1:
		icon = cv2.resize(icon, None, fx=r, fy=r)

	############## アイコンの余白削除と☆・属性の判定 ##############
	# マッチング
	frames = [
		imread(resource_path(f'resource\\frame_{i}_{j}.png'), -1) for i, j in [(3, 1), (3, 2), (4, 1), (4, 2)]
	]
	results = []
	for f in frames:
		color, mask = unpack_rgba(f)
		mask = np.ones_like(mask, dtype=np.uint8) * mask
		res = scaledMatchTemplateSearch(icon, color, floatrange(0.6, 1, 0.01), mask=mask)
		results.append(res)

	idx = np.argmax(results, axis=0)[1]
	frame = frames[idx]
	match_result = results[idx]
	scale, score, x1, y1 = match_result

	# アイコン領域の切り出し
	s = min(icon.shape[:2])
	x1, y1 = int(x1), int(y1)
	x2, y2 = int(x1 + s*scale), int(y1 + s*scale)
	# print(scale, x1, y1, x2, y2)

	icon = icon[y1:y2, x1:x2, :]

	# アイコンを中央50%にクロップ (☆、鍵、特効マーク等を除くため)
	h, w, _ = icon.shape
	crop_min = int(h * 0.25)
	crop_max = int(h * 0.75)
	icon = icon[crop_min:crop_max, crop_min:crop_max, :]


	############## マッチング ##############

	# 1%単位で探索
	"""
	scale, score, x1, y1 = scaledMatchTemplateSearch(photo, icon, floatrange(0.01, 1+0.01, 0.01))

	## 0.05%単位で探索
	scale, score, x1, y1 = scaledMatchTemplateSearch(photo, icon, floatrange(scale-0.01, scale+0.01, 0.0005))
	# # print(best_result)
	"""
	scale, score, x1, y1 = twoStepScaledMatchTemplateSearch(photo, icon, 0.1, 1)

	s = min(photo.shape[:2])
	x1, y1 = int(x1), int(y1)
	x2, y2 = int(x1 + s*scale), int(y1 + s*scale)
	# print(x1, y1, x2, y2)

	# フォトからアイコン領域を切り出す
	center = ((x2 + x1) // 2, (y2 + y1) // 2)
	half_size = (x2 - x1)
	size = half_size * 2
	photo_crop = photo[center[1] - half_size:center[1] + half_size, center[0] - half_size:center[0] + half_size, :]
	# print(center[1] - half_size, center[1] + half_size, center[0] - half_size, center[0] + half_size)

	# 枠をつける
	frame = cv2.resize(frame, (size, size))
	frame_color, frame_mask = unpack_rgba(frame)

	result = photo_crop
	result[frame_mask, :] = frame_color[frame_mask, :]
	result = cv2.resize(result, (240, 240))


	# cv2.imshow('test', icon)
	# cv2.imshow('test', photo)
	# cv2.imshow('test', result)
	# while True:
	# 	if cv2.waitKey(2) == 27:
	# 		cv2.destroyAllWindows()
	# 		break

	imwrite('out.png', result)
	subprocess.run('PAUSE', shell=True)