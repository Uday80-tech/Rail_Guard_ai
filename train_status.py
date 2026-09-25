import cv2
import numpy as np


class TrainStatus:

    def __init__(self):

        # Previous grayscale frame
        self.prev_gray = None

        # Previous train bounding box
        self.prev_bbox = None

        # Previous train ID
        self.last_track_id = None

        # Scores
        self.moving_score = 0
        self.stopped_score = 0

        # Final status
        self.status = "CHECKING"

        # =========================================
        # Optical Flow thresholds
        # =========================================

        self.FLOW_THRESHOLD = 1.0

        self.MOVING_PERCENT = 5.0
        self.STOPPED_PERCENT = 3.0

        self.AVG_MOTION_MOVING = 0.30
        self.AVG_MOTION_STOPPED = 0.20

        # =========================================
        # Score settings
        # =========================================

        self.MAX_SCORE = 10
        self.STATUS_SCORE = 5

        # =========================================
        # Processing resolution
        # =========================================

        self.PROCESS_WIDTH = 640


    def reset(self):

        self.prev_gray = None
        self.prev_bbox = None

        self.moving_score = 0
        self.stopped_score = 0

        self.status = "CHECKING"


    def resize_bbox(self, bbox, original_width, original_height,
                    new_width, new_height):

        """
        Convert original-frame bbox into resized-frame coordinates.
        """

        x1, y1, x2, y2 = bbox

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        return x1, y1, x2, y2


    def get_intersection(self, bbox1, bbox2):

        """
        Get common region between previous and current
        train bounding boxes.
        """

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])

        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        return x1, y1, x2, y2


    def update(self, frame, bbox, track_id):

        """
        frame    -> original video frame
        bbox     -> (x1, y1, x2, y2)
        track_id -> ByteTrack train ID
        """

        # =========================================
        # 1. Check train ID
        # =========================================

        if self.last_track_id != track_id:

            self.reset()

            self.last_track_id = track_id


        # =========================================
        # 2. Original frame dimensions
        # =========================================

        original_height, original_width = frame.shape[:2]


        # =========================================
        # 3. Resize frame only for optical flow
        # =========================================

        scale = self.PROCESS_WIDTH / original_width

        process_height = int(
            original_height * scale
        )

        small_frame = cv2.resize(
            frame,
            (self.PROCESS_WIDTH, process_height)
        )


        # =========================================
        # 4. Convert to grayscale
        # =========================================

        gray = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2GRAY
        )


        # =========================================
        # 5. Convert train bbox to resized frame
        # =========================================

        current_bbox = self.resize_bbox(
            bbox,
            original_width,
            original_height,
            self.PROCESS_WIDTH,
            process_height
        )


        x1, y1, x2, y2 = current_bbox


        # Keep bbox inside frame

        x1 = max(0, min(x1, self.PROCESS_WIDTH - 1))
        y1 = max(0, min(y1, process_height - 1))

        x2 = max(0, min(x2, self.PROCESS_WIDTH))
        y2 = max(0, min(y2, process_height))


        if x2 <= x1 or y2 <= y1:

            return self.status


        current_bbox = (x1, y1, x2, y2)


        # =========================================
        # 6. First frame
        # =========================================

        if self.prev_gray is None:

            self.prev_gray = gray.copy()
            self.prev_bbox = current_bbox

            return self.status


        # =========================================
        # 7. Optical Flow on FULL FRAME
        # =========================================

        flow = cv2.calcOpticalFlowFarneback(

            self.prev_gray,
            gray,

            None,

            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )


        # =========================================
        # 8. Get motion magnitude
        # =========================================

        magnitude = cv2.magnitude(
            flow[..., 0],
            flow[..., 1]
        )


        # =========================================
        # 9. Find common train region
        # =========================================

        rx1, ry1, rx2, ry2 = self.get_intersection(
            self.prev_bbox,
            current_bbox
        )


        if rx2 <= rx1 or ry2 <= ry1:

            self.prev_gray = gray.copy()
            self.prev_bbox = current_bbox

            return self.status


        # =========================================
        # 10. Remove bbox boundary
        # =========================================

        region_width = rx2 - rx1
        region_height = ry2 - ry1

        margin_x = int(region_width * 0.10)
        margin_y = int(region_height * 0.10)


        rx1 += margin_x
        ry1 += margin_y

        rx2 -= margin_x
        ry2 -= margin_y


        if rx2 <= rx1 or ry2 <= ry1:

            self.prev_gray = gray.copy()
            self.prev_bbox = current_bbox

            return self.status


        # =========================================
        # 11. Extract train motion
        # =========================================

        train_magnitude = magnitude[
            ry1:ry2,
            rx1:rx2
        ]


        if train_magnitude.size == 0:

            self.prev_gray = gray.copy()
            self.prev_bbox = current_bbox

            return self.status


        # =========================================
        # 12. Calculate moving pixels
        # =========================================

        moving_pixels = (
            train_magnitude > self.FLOW_THRESHOLD
        )


        moving_percentage = (

            np.count_nonzero(moving_pixels)
            /
            moving_pixels.size

        ) * 100


        # =========================================
        # 13. Average motion
        # =========================================

        average_motion = np.mean(
            train_magnitude
        )


        # =========================================
        # 14. Movement evidence
        # =========================================

        moving_evidence = (

            moving_percentage > self.MOVING_PERCENT

            and

            average_motion > self.AVG_MOTION_MOVING
        )


        # =========================================
        # 15. Stopped evidence
        # =========================================

        stopped_evidence = (

            moving_percentage < self.STOPPED_PERCENT

            and

            average_motion < self.AVG_MOTION_STOPPED
        )


        # =========================================
        # 16. Update scores
        # =========================================

        if moving_evidence:

            self.moving_score += 1

            self.stopped_score = max(
                0,
                self.stopped_score - 1
            )


        elif stopped_evidence:

            self.stopped_score += 1

            self.moving_score = max(
                0,
                self.moving_score - 1
            )


        else:

            self.moving_score = max(
                0,
                self.moving_score - 0.5
            )

            self.stopped_score = max(
                0,
                self.stopped_score - 0.5
            )


        # =========================================
        # 17. Limit scores
        # =========================================

        self.moving_score = min(
            self.moving_score,
            self.MAX_SCORE
        )

        self.stopped_score = min(
            self.stopped_score,
            self.MAX_SCORE
        )


        # =========================================
        # 18. Final status
        # =========================================

        if self.moving_score >= self.STATUS_SCORE:

            self.status = "MOVING"


        elif self.stopped_score >= self.STATUS_SCORE:

            self.status = "STOPPED"


        # =========================================
        # 19. Save current frame and bbox
        # =========================================

        self.prev_gray = gray.copy()
        self.prev_bbox = current_bbox


        # =========================================
        # 20. Return status
        # =========================================

        return self.status


# =================================================
# Create detector
# =================================================

train_status_detector = TrainStatus()


# =================================================
# Function used by main.py
# =================================================

def check_train_status(frame, bbox, train_id):

    return train_status_detector.update(
        frame,
        bbox,
        train_id
    )