'''
Test deflicker/__init__.py
'''

import pytest 
import numpy as np

from deflicker.__init__ import calc_brightness, rolling_mean, scale_image_brightness

base_url = "https://raw.githubusercontent.com/yavuzceliker/sample-images/refs/heads/main/docs/"
urls = [f"{base_url}image-111{i}.jpg" for i in range(3)]


def test_calc_brightness_static():
    brightness = calc_brightness(urls)
    assert brightness is not None
    assert all(isinstance(b, float) for b in brightness)
    assert len(brightness) == len(urls)
    expected_brightness = [125.0928, 139.1198, 128.8109]
    assert brightness == pytest.approx(expected_brightness, abs=0.0001)

def test_scale_image_brightness_static():
    array = np.full((100, 100, 3), 100, dtype=np.uint8)
    scaled = scale_image_brightness(array, 1.5)
    assert np.array_equal(scaled, np.full((100, 100, 3), 150, dtype=np.uint8))

    # Test overflow
    array = np.full((100, 100, 3), 100, dtype=np.uint8)
    scaled = scale_image_brightness(array, 3)
    assert np.array_equal(scaled, np.full((100, 100, 3), 255, dtype=np.uint8))

    # Test image with alpha channel
    array = np.full((1000, 1000, 4), 100, dtype=np.uint8)
    scaled = scale_image_brightness(array, 1.5)
    assert np.array_equal(scaled, np.full((1000, 1000, 4), 150, dtype=np.uint8))

    # Test 16-bit dng image
    array = np.full((3000, 2000, 3), 10000, dtype=np.uint16)
    scaled = scale_image_brightness(array, 1.5)
    assert np.array_equal(scaled, np.full((3000, 2000, 3), 15000, dtype=np.uint16))

def test_rolling_mean_static():
    # Length of 10
    brightness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    target_brightness = rolling_mean(brightness, 2)
    assert len(target_brightness) == len(brightness)
    expected_target_brightness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan], dtype=np.float32)
    assert np.array_equal(target_brightness, expected_target_brightness, equal_nan=True)

    target_brightness = rolling_mean(brightness, 3)
    assert len(target_brightness) == len(brightness)
    expected_target_brightness = np.array([np.nan, 1, 1, 1, 1, 1, 1, 1, 1, np.nan], dtype=np.float32)
    assert np.array_equal(target_brightness, expected_target_brightness, equal_nan=True)

    # Length of 9
    brightness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    target_brightness = rolling_mean(brightness, 2)
    assert len(target_brightness) == len(brightness)
    expected_target_brightness = np.array([1, 1, 1, 1, 1, 1, 1, 1, np.nan], dtype=np.float32)
    assert np.array_equal(target_brightness, expected_target_brightness, equal_nan=True)

    target_brightness = rolling_mean(brightness, 3)
    assert len(target_brightness) == len(brightness)
    expected_target_brightness = np.array([np.nan, 1, 1, 1, 1, 1, 1, 1, np.nan], dtype=np.float32)
    assert np.array_equal(target_brightness, expected_target_brightness, equal_nan=True)

    # Square function
    brightness = np.array([0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0])
    target_brightness = rolling_mean(brightness, 5)
    assert len(target_brightness) == len(brightness)
    expected_target_brightness = np.array([np.nan, np.nan, 0, 2, 4, 6, 6, 6, 4, 2, 0, np.nan, np.nan], dtype=np.float32)
    assert np.array_equal(target_brightness, expected_target_brightness, equal_nan=True)

    # Too large of window
    target_brightness = rolling_mean(brightness, 20)
    assert len(target_brightness) == len(brightness)
    expected_target_brightness = np.array([np.nan, np.nan, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, np.nan, np.nan, np.nan], dtype=np.float32)
    assert np.array_equal(target_brightness, expected_target_brightness, equal_nan=True)
