# noise_filter.py

def filter_noise_rectangles(frame, rects, left_ignore_pct=0.2, right_ignore_pct=0.2, top_ignore_pct=0.2):
    """
    Filters out detections (rectangles) that overlap the left, right, or top margins of the frame.

    Parameters:
        frame: The current video frame as a NumPy array.
        rects: List of tuples (x, y, w, h) representing bounding boxes.
        left_ignore_pct: Fraction of the frame's width at the left to ignore.
        right_ignore_pct: Fraction of the frame's width at the right to ignore.
        top_ignore_pct: Fraction of the frame's height at the top to ignore.

    Returns:
        A list of rectangles that do not overlap any of the ignore zones.
    """
    height, width = frame.shape[:2]
    
    # Define ignore zones covering the entire height (for left and right) and entire width (for top)
    left_zone = (0, 0, int(width * left_ignore_pct), height)
    right_zone = (int(width - width * right_ignore_pct), 0, int(width * right_ignore_pct), height)
    top_zone = (0, 0, width, int(height * top_ignore_pct))
    
    def overlaps(rect, zone):
        x, y, w, h = rect
        zx, zy, zw, zh = zone
        rect_right, rect_bottom = x + w, y + h
        zone_right, zone_bottom = zx + zw, zy + zh
        
        # Check if the two rectangles overlap:
        if rect_right <= zx or x >= zone_right:
            return False
        if rect_bottom <= zy or y >= zone_bottom:
            return False
        return True

    filtered_rects = []
    for rect in rects:
        # If the rectangle overlaps any of the ignore zones, skip it
        if overlaps(rect, left_zone) or overlaps(rect, right_zone) or overlaps(rect, top_zone):
            continue
        filtered_rects.append(rect)
    
    return filtered_rects
