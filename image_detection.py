import cv2
import numpy as np

# Define HSV ranges for each color we care about.
# These need to be tuned so that they roughly match the hues of the colors
COLOR_RANGES = {
	"red": [
		# Red wraps around the hue axis, so we use two ranges.
		((0, 120, 70), (10, 255, 255)),
		((170, 120, 70), (180, 255, 255)),
	],
	"green": [
		((35, 80, 70), (85, 255, 255)),
	],
	"blue": [
		((90, 80, 70), (130, 255, 255)),
	],
	# Add more colors if we need...
}

def detect_colored_cups(image_bgr, min_area=500):
	"""
	image_bgr: OpenCV BGR image (uint8, shape HxWx3)
	min_area: minimum contour area to accept as a cup (in pixels)
	
	Returns: dict: {"red":(cx, cy), "blue":(cx, cy)}
	"""
	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	results = {color: [] for color in COLOR_RANGES}  # initialize dict

	for color_name, ranges in COLOR_RANGES.items():
		mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

		# combine masks if multiple ranges exist (like red wraparound)
		for lower, upper in ranges:
			mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

		# cleanup noise
		kernel = np.ones((5, 5), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		# find contour blobs
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			if cv2.contourArea(cnt) < min_area:
				continue

			M = cv2.moments(cnt)
			if M["m00"] == 0:
				continue

			cx = int(M["m10"] / M["m00"])
			cy = int(M["m01"] / M["m00"])
			results[color_name].append((cx, cy))
	return results