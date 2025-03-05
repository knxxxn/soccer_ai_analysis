from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.track_history = defaultdict(lambda: {
            "position": None,
            "bbox": None,
            "color": None,
            "last_seen": 0,
            "last_consistent_frame": 0
        })
        self.previous_ball_owner = None
        self.previous_team = None
        self.commentary = ""

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # 누락된 값 보간
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Supervision Detection 형식으로 변환
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # 골키퍼를 플레이어 객체로 변환
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # 객체 추적
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def update_ball_owner(self, player_id, team_id):
        if self.previous_ball_owner is None:
            self.commentary = f"Player {player_id} has the ball.\n플레이어 {player_id}가 공을 가지고 있습니다."
        elif player_id != self.previous_ball_owner:
            if team_id != self.previous_team:
                self.commentary = f"Player {self.previous_ball_owner} lost the ball. Player {player_id} now has it. \n플레이어 {self.previous_ball_owner} 공을 뺐겼습니다. 플레이어 {player_id}가 공을 가지고 있습니다."
            else:
                self.commentary = f"Player {self.previous_ball_owner} passed to Player {player_id}. \n플레이어 {self.previous_ball_owner}가 플레이어 {player_id}에게 패스를 했습니다."
        self.previous_ball_owner = player_id
        self.previous_team = team_id

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = team_2 = 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        return frame

    def draw_subtitle(self, frame, subtitle_text, position_offset=100):
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # 사용자 폰트 경로 설정
        font_path = r"C:/Users/USER/Documents/GitHub/sports_analysis/Noto_Sans_KR/static/NotoSansKR-Regular.ttf"
        font = ImageFont.truetype(font_path, 32)

        frame_height = frame.shape[0]
        position = (50, frame_height - position_offset)

        draw.text(position, subtitle_text, font=font, fill=(255, 255, 255, 255))

        frame = np.array(img_pil.convert('RGB'))

        return frame

    def draw_event_subtitle(self, frame, event_text, position_offset=100):
        if not event_text:
            return frame

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        # 사용자 폰트 경로 설정
        font_path = r"C:/Users/USER/Documents/GitHub/sports_analysis/Noto_Sans_KR/static/NotoSansKR-Regular.ttf"
        font = ImageFont.truetype(font_path, 32)

        frame_height = frame.shape[0]
        position = (50, frame_height - position_offset)

        draw.text(position, event_text, font=font, fill=(255, 255, 0, 255))

        frame = np.array(img_pil.convert('RGB'))

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, subtitle_texts, event_texts):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, np.array(team_ball_control))
            frame = self.draw_subtitle(frame, subtitle_texts[frame_num], position_offset=150)  # 자막 위치 설정
            frame = self.draw_event_subtitle(frame, event_texts[frame_num], position_offset=200)  # 이벤트 자막 위치 설정

            output_video_frames.append(frame)

        return output_video_frames
