'''	COMPSCI 373 (2021) - University of Auckland
	ASSIGNMENT ONE - QR Code Detection
	Simon Shan  441147157

	[!] * * *   PLEASE NOTE   * * * [!]
	In this code, sobel and mean kernel operations may look weird.
	They are optimised for speed.
	For example, instead of [1 0 -1] * [a b c] = [1*a, 0*b, -1*c],
	it is simply a - c in my representation
'''

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import imageIO.png

# custom decorator for timing functions
from time_this import time_this


#########################  settings  #########################
SAVE_IMAGES = True      # writes outputs to PNG files        #
PLOT_IMAGES = True      # plots outputs in matplotlib window #
#########################  settings  #########################




#######################  image file IO  ########################

@time_this
def read_rgb_triplets(filename):
	'''each pixel is an (r,g,b) tuple'''
	width, height, rgb_array, _ = imageIO.png.Reader(filename=filename).read()
	print(f'[ read  {filename}, {width}w x {height}h ]')

	rgb_triplets = [list(zip(*(iter(row),) * 3)) for row in rgb_array]
	return rgb_triplets, width, height

@time_this
def write_greyscale(filename, pixels):
	try:
		pixels = contrast_stretch(pixels)
		with open(filename, 'wb') as file:
			imageIO.png.Writer(width, height, greyscale=True).write(file, pixels)
		print(f'[ wrote {filename}, {width}w x {height}h [greyscale] ]')
	except:
		print(f'[ERROR] failed to write {filename}')



###################  processing operations  ####################

@time_this
def rgb_to_greyscale(rgb_triplets):
	return [ [round(.299*px[0] + .587*px[1] + .114*px[2])
					for px  in row] for row in rgb_triplets ]

# @time_this	# timing this function clutters the terminal
def contrast_stretch(image):
	'''accepts only greyscale images'''
	flat = [px for row in image for px in row]
	pixel_min = min(flat)
	pixel_max = max(flat)

	gain = 255 / (pixel_max-pixel_min)
	bias = -pixel_min * gain

	if gain==1 and bias==0: return image
	return [ [round(px*gain+bias) for px in row] for row in image ]

def pad_0(image):
	'''accepts greyscale or binary images'''
	dy = (width -len(image[0]))//2
	dx = (height-len(image   ))//2
	copy  = [ [0]*(width) ] * dx
	copy += [ [0]*dy + row + [0]*dy for row in image ]
	copy += [ [0]*(width) ] * dx
	return copy

@time_this
def sobel_horizontal(image):
	'''	[ 1  2  1]             [ 1]
		[ 0  0  0] = [1 2 1] x [ 0] x (1/4)
		[-1  0 -2]             [-1]
		    ^                 ^
	  9 calculations    3 calculations
	'''
	# * [1 0 -1]^T operation
	copy = [ [image[i][j]-image[i+2][j] for j in range(width)] for i in range(height-2) ]
	# * [1 2 1] * (1/4) operation
	copy = [ [(copy[i][j]+2*copy[i][j+1]+copy[i][j+2])/4 for j in range(width-2)] for i in range(height-2) ]
	return pad_0(copy)

@time_this
def sobel_vertical(image):
	'''	[1  0 -1]               [1]
		[2  0 -2] = [1  0 -1] x [2] x (1/4)
		[1  0 -1]               [1]
		    ^                 ^
	  9 calculations    3 calculations
	'''
	# [1  0 -1] operation
	copy = [ [image[i][j]-image[i][j+2] for j in range(width-2)] for i in range(height) ]
	# [1 2 1]^T * (1/4) operation
	copy = [ [(copy[i][j]+2*copy[i+1][j]+copy[i+2][j])/4 for j in range(width-2)] for i in range(height-2) ]
	return pad_0(copy)

@time_this
def combine(edges_h, edges_v):
	# using the appromimate formula gm = |gx(x,y)| + |gy(x,y)|
	# it is faster than the formula gm = sqrt( gx(x,y)^2 + gx(x,y)^2 )
	copy = [ [abs(edges_h[i][j])+abs(edges_v[i][j]) for j in range(width )] for i in range(height)]
	return [ [ 255 if px>255 else round(px) for px in row] for row in copy ]

@time_this
def mean_smoothing(image):
	'''	smoothing using 5x5 mean kernel
		5x5 kernel requires less passes than 3x3 kernel
			[1 1 1 1 1]                 [1]
			[1 1 1 1 1]                 [1]
			[1 1 1 1 1] = [1 1 1 1 1] x [1] x (1/25)
			[1 1 1 1 1]                 [1]
			[1 1 1 1 1]                 [1]
			     ^                    ^
		   25 calculations      3 calculations
	'''
	# sorry this line is long but speed requires
	# [1 1 1 1 1]^T operation
	copy = [ [image[i][j]+image[i+1][j]+image[i+2][j]+image[i+3][j]+image[i+4][j] for j in range(width)] for i in range(height-4) ]
	# [1 1 1 1 1] * (1/25) operation
	copy = [ [round(sum(row[j:j+5])/25) for j in range(width-4)] for row in copy ]
	# 2 paddings
	return pad_0(pad_0(copy))

@time_this
def thresholding(image, threshold):
	return [ [ 0 if px<threshold else 255 for px in row ] for row in image ]

@time_this
def dilation(image):
	copy = [[0]*width for _ in range(height)]
	for i in range(height-2):
		for j in range(width-2):
			if image[i+1][j+1]:
				copy[i  ][j:j+3] = [255, 255, 255]
				copy[i+1][j:j+3] = [255, 255, 255]
				copy[i+2][j:j+3] = [255, 255, 255]
	return copy

@time_this
def erosion(image):
	copy = [[0]*width for _ in range(height)]
	for i in range(height-2):
		for j in range(width-2):
			window = image[i][j:j+3]+image[i+1][j:j+3]+image[i+2][j:j+3]
			copy[i+1][j+1] = 255 if all(window) else 0
	return copy

# @time_this	unnecessary, dilation and erotion are already timed
def closing(image):
	copy = dilation(image)
	return erosion (copy )

@time_this
def grass_fire(binary):
	mapping = [ [0]*width for _ in range(height) ]
	label, label_count = 0, {}

	for i in range(height):
		for j in range(width):
			if not binary[i][j] or mapping[i][j]: continue

			label += 1
			label_count[label] = 0
			px = [(i, j)]

			while px:
				_i, _j = px.pop(0)
				mapping[_i][_j] = label
				label_count[label] += 1
				if 0<=_i-1<height and 0<=_j<width and binary[_i-1][_j] and not mapping[_i-1][_j] and (_i-1,_j) not in px:
					px.append((_i-1,_j))
				if 0<=_i+1<height and 0<=_j<width and binary[_i+1][_j] and not mapping[_i+1][_j] and (_i+1,_j) not in px:
					px.append((_i+1,_j))
				if 0<=_i<height and 0<=_j-1<width and binary[_i][_j-1] and not mapping[_i][_j-1] and (_i,_j-1) not in px:
					px.append((_i,_j-1))
				if 0<=_i<height and 0<=_j+1<width and binary[_i][_j+1] and not mapping[_i][_j+1] and (_i,_j+1) not in px:
					px.append((_i,_j+1))

	largest_label = max(label_count, key=label_count.get)
	return [ [255 if label==largest_label else 0 for label in row] for row in mapping ]

'''	[!] * * *   DEPRECATED   * * * [!] '''
# @time_this
# def extract_corners(image):
# 	left, right  = (width, height), (0, 0)
# 	top , bottom = (width, height), (0, 0)
# 	for i in range(height):
# 		for j in range(width):
# 			if not image[i][j]: continue
# 			if j <=   left[0]: left   = (j, i)
# 			if j >   right[0]: right  = (j, i)
# 			if i <     top[1]: top    = (j, i)
# 			if i >= bottom[1]: bottom = (j, i)
# 	return top, left, bottom, right

@time_this
def extract_centre(image):
	x_min, y_min, x_max, y_max = width, height, 0, 0
	for y in range(height):
		for x in range(width):
			if not image[y][x]: continue
			if x < x_min: x_min = x
			if y < y_min: y_min = y
			if x > x_max: x_max = x
			if y > y_max: y_max = y
	return (x_max+x_min)//2, (y_max+y_min)//2

@time_this
def quarter_corners(image):
	cx, cy = extract_centre(image)
	corners = [(), (), (), ()]
	d_max = 0
	for y in range(cy):
		for x in range(cx):
			if not image[y][x]: continue
			d = ((cx-x)**2 + (cy-y)**2)**0.5
			if d > d_max:
				d_max = d
				corners[0] = (x, y)
	d_max = 0
	for y in range(cy):
		for x in range(cx, width):
			if not image[y][x]: continue
			d = ((cx-x)**2 + (cy-y)**2)**0.5
			if d > d_max:
				d_max = d
				corners[1] = (x, y)
	d_max = 0
	for y in range(cy, height):
		for x in range(cx, width):
			if not image[y][x]: continue
			d = ((cx-x)**2 + (cy-y)**2)**0.5
			if d > d_max:
				d_max = d
				corners[2] = (x, y)
	d_max = 0
	for y in range(cy, height):
		for x in range(cx):
			if not image[y][x]: continue
			d = ((cx-x)**2 + (cy-y)**2)**0.5
			if d > d_max:
				d_max = d
				corners[3] = (x, y)

	return (cx, cy), corners



##########################  plotting  ##########################

def subplot(image, step, title=''):
	if PLOT_IMAGES:
		plt.subplot(3, 3, step)
		plt.axis('off')
		plt.imshow(image, cmap='gray')
		plt.title(f'step{step} - title', size=6, family='monospace')

	if SAVE_IMAGES and step!=9:
		write_greyscale(f'output/step{step}.png', image)

def draw_features(centre, points):
	plt.plot(*centre, 'mo')
	rect = Polygon( points, linewidth=3, edgecolor='g', facecolor='none' )
	plt.gca().add_patch(rect)
	for point, colour in zip(points, ['ro', 'yo', 'co', 'bo']):
		plt.plot(*point, colour)



#########################  detection  ##########################

def detect_QR_code(filename):
	plt.figure('Steps 1-9', dpi=120)
	plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)

	global width, height
	rgb_triplets, width, height = read_rgb_triplets(filename)

	# Step 1 - convert to greyscale #
	greyscale = rgb_to_greyscale(rgb_triplets)
	greyscale = contrast_stretch(greyscale)
	subplot(greyscale, step=1, title='greyscale')

	# Step 2 - compute horizontal edges #
	edges_h = sobel_horizontal(greyscale)
	subplot(edges_h, 2, 'horizontal edges')

	# Step 3 - compute vertical edges #
	edges_v = sobel_vertical  (greyscale)
	subplot(edges_v, 3, 'vertical edges')
	
	# Step 4 - compute gradient magnitude #
	image = combine(edges_h, edges_v)
	image = contrast_stretch(image)
	subplot(image, 4, 'gradient magnitude')

	# Step 5 - smoothing over #
	image = mean_smoothing(image)
	image = mean_smoothing(image)
	image = contrast_stretch(image)
	subplot(image, 5, 'mean smoothing')

	# Step 6 - thresholding #
	image = thresholding(image, 50)
	subplot(image, 6, 'thresholding')

	# Step 7 - close holes #
	image = closing(image)
	subplot(image, 7, 'closing')


	# Step 8 - perform connect component analysis #
	image = grass_fire(image)
	subplot(image, 8, 'largest component')

	# Step 9 - extract box #
	centre, corners = quarter_corners(image)
	print('centre:', centre, 'corners:', *corners, sep=' ')
	subplot(rgb_triplets, 9, 'bounding box')
	draw_features(centre, corners)

#########   [ Final Output ]   #########

	plt.figure('[ Final Output ]')
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.axis('off')

	plt.imshow(rgb_triplets, cmap='gray')
	draw_features(centre, corners)

	plt.savefig('output/step9.png')
	plt.show()


################################################################
####                    MAIN STARTS HERE                    ####
################################################################
if __name__ == '__main__':
	# filename = "images/poster1small.png"
	filename = "images/bloomfield.png"
	# filename = "images/connecticut.png"
	# filename = "images/poster1smallrotated.png"
	# filename = "images/shanghai.png"
	detect_QR_code(filename)
