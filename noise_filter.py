# noise_filter.py

def filter_noise_rectangles(frame, rects, left_ignore_pct=0.2, right_ignore_pct=0.2, top_ignore_pct=0.2, full_top_ignore_pct=0.2):
    """
    Filters out rectangles (detections) that fall into noisy regions:
      - The top-left corner (based on left_ignore_pct and top_ignore_pct)
      - The top-right corner (based on right_ignore_pct and top_ignore_pct)
      - The entire top region of the frame (based on full_top_ignore_pct)
    
    Parameters:
        frame: The current video frame as a NumPy array.
        rects: List of tuples (x, y, w, h) representing bounding boxes.
        left_ignore_pct: Fraction of frame width to ignore in the top-left area.
        right_ignore_pct: Fraction of frame width to ignore in the top-right area.
        top_ignore_pct: Fraction of frame height to ignore for top-left and top-right zones.
        full_top_ignore_pct: Fraction of frame height from the top to ignore for all detections.
        
    Returns:
        List of rectangles not overlapping any of the ignore zones.
    """
    height, width = frame.shape[:2]
    
    # Define ignore zones for top-left and top-right corners
    tl_zone = (0, 0, int(width * left_ignore_pct), int(height * top_ignore_pct))
    tr_zone = (int(width - width * right_ignore_pct), 0, int(width * right_ignore_pct), int(height * top_ignore_pct))
    
    # Define the full top ignore zone (across the entire width)
    full_top_zone = (0, 0, width, int(height * full_top_ignore_pct))
    
    def overlaps(rect, zone):
        x, y, w, h = rect
        zx, zy, zw, zh = zone
        rect_right, rect_bottom = x + w, y + h
        zone_right, zone_bottom = zx + zw, zy + zh
        
        # Check for horizontal and vertical overlap
        if x >= zone_right or zx >= rect_right:
            return False
        if y >= zone_bottom or zy >= rect_bottom:
            return False
        return True

    filtered_rects = []
    for rect in rects:
        if overlaps(rect, tl_zone) or overlaps(rect, tr_zone) or overlaps(rect, full_top_zone):
            continue
        filtered_rects.append(rect)
    
    return filtered_rects
