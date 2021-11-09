# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from adet.utils.visualizer import TextVisualizer
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    # dictionary = 'aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ'
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D‑", "d‑"]


def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char

def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition

class SubmissionDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def _decode_recognition(self, rec):
        #        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        # CTLABELS = ['^', '\\', '}', 'ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', ' ', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']
        CTLABELS = [
            " ",
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "{",
            "|",
            "}",
            "~",
            "ˋ",
            "ˊ",
            "﹒",
            "ˀ",
            "˜",
            "ˇ",
            "ˆ",
            "˒",
            "‑",
        ]

        s = ""
        for c in rec:
            c = int(c)
            if c < 104:
                s += CTLABELS[c]
            elif c == 104:
                s += u"口"
        return decoder(s)
    
    def _bezier_to_poly(self, bezier):
        # bezier to polygon
        u = np.linspace(0, 1, 20)
        bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
        points = (
            np.outer((1 - u) ** 3, bezier[:, 0])
            + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1])
            + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2])
            + np.outer(u ** 3, bezier[:, 3])
        )
        points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
        
        return points

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        outputs = self.predictor(image)

        # bbox_ltrbs = outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        recs = outputs['instances'].recs
        scores = outputs['instances'].scores.tolist()
        beziers = outputs['instances'].beziers.cpu().numpy()
        
        results = []
        for bezier, rec, score in zip(beziers, recs, scores):
            text = self._decode_recognition(rec)
            polygon = self._bezier_to_poly(bezier)
            ele = []
            for i in [0,19,20,39]:
                ele.append(polygon[i][0])
                ele.append(polygon[i][1])
            ele.append(score)
            ele.append(text)
            # ele.append(polygon)
            results.append(ele)

        return results

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def vis_bases(self, bases):
        basis_colors = [[2, 200, 255], [107, 220, 255], [30, 200, 255], [60, 220, 255]]
        bases = bases[0].squeeze()
        bases = (bases / 8).tanh().cpu().numpy()
        num_bases = len(bases)
        fig, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
        for i, basis in enumerate(bases):
            basis = (basis + 1) / 2
            basis = basis / basis.max()
            basis_viz = np.zeros((basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
            basis_viz[:, :, 0] = basis_colors[i][0]
            basis_viz[:, :, 1] = basis_colors[i][1]
            basis_viz[:, :, 2] = np.uint8(basis * 255)
            basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
            axes[i // 2][i % 2].imshow(basis_viz)
        plt.show()

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        # video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
        # video_visualizer = TextVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                # vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
                video_visualizer = TextVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
                vis_frame = video_visualizer.draw_instance_predictions(predictions=predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue))

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
