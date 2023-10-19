import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

# Define the architecture of the Faster R-CNN model
class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes, anchor_scales, anchor_ratios, image_height, image_width):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(image_height, image_width, 3))
        self.rpn = RPN(anchor_scales, anchor_ratios, feature_stride=1)
        self.classifier = Classifier(num_classes)

    def call(self, inputs, training=False):
        features = self.backbone(inputs)
        rpn_output = self.rpn(features, training=training)
        detection_boxes, detection_scores = self.classifier(features, rpn_output, training=training)
        return detection_boxes, detection_scores

# Implement the Region Proposal Network
class RPN(tf.keras.layers.Layer):
    def __init__(self, anchor_scales, anchor_ratios, feature_stride=1):
        super(RPN, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.feature_stride = feature_stride
        self.conv = layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.cls_head = layers.Conv2D(2 * len(anchor_scales), (1, 1), activation='linear')
        self.reg_head = layers.Conv2D(4 * len(anchor_scales), (1, 1), activation='linear')


    def call(self, features, training=False):
        x = self.conv(features)
        class_logits = self.cls_head(x)
        class_logits = tf.reshape(class_logits, (tf.shape(class_logits)[0], -1, 2))
        bounding_box_deltas = self.reg_head(x)
        bounding_box_deltas = tf.reshape(bounding_box_deltas, (tf.shape(bounding_box_deltas)[0], -1, 4))
        
        if training:
            return class_logits, bounding_box_deltas
        else:
            rpn_boxes, rpn_scores, rpn_anchors = self.generate_anchors(class_logits, bounding_box_deltas)
            return rpn_boxes, rpn_scores, rpn_anchors

    def generate_anchors(self, class_logits, bounding_box_deltas):
        # Get the shape of the feature map
        feature_map_shape = tf.shape(class_logits)[1:3]

        # Compute the scales and ratios of anchors
        scales, ratios = np.meshgrid(self.anchor_scales, self.anchor_ratios)
        scales, ratios = scales.flatten(), ratios.flatten()

        # Calculate the width and height of the anchor boxes
        widths = scales * np.sqrt(ratios)
        heights = scales / np.sqrt(ratios)

        # Calculate the center coordinates of the anchor boxes
        x_centers = np.arange(0, feature_map_shape[1]) * self.feature_stride
        y_centers = np.arange(0, feature_map_shape[0]) * self.feature_stride
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)
        x_centers, y_centers = x_centers.flatten(), y_centers.flatten()

        # Use bounding box deltas to refine the anchor boxes
        widths = tf.exp(bounding_box_deltas[:, :, 2]) * widths
        heights = tf.exp(bounding_box_deltas[:, :, 3]) * heights
        x_centers = x_centers + bounding_box_deltas[:, :, 0] * self.feature_stride
        y_centers = y_centers + bounding_box_deltas[:, :, 1] * self.feature_stride

        # Calculate the anchor boxes' coordinates
        anchor_x1 = x_centers - 0.5 * widths
        anchor_x2 = x_centers + 0.5 * widths
        anchor_y1 = y_centers - 0.5 * heights
        anchor_y2 = y_centers + 0.5 * heights

        # Calculate the number of anchors
        num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        # Reshape the coordinates of anchor boxes
        anchor_boxes = tf.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], axis=-1)

        # Expand dimensions to match batch size
        anchor_boxes = tf.expand_dims(anchor_boxes, axis=0)

        # Expand dimensions for each anchor
        anchor_boxes = tf.tile(anchor_boxes, [tf.shape(class_logits)[0], 1, 1])

        return anchor_boxes

class Classifier(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.roi_pooling = layers.MaxPooling2D(pool_size=(7, 7))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu')
        self.cls_head = layers.Dense(num_classes, activation='softmax')
        self.reg_head = layers.Dense(4, activation='linear')

    def generate_rois(self, rpn_output, threshold=0.7):
        rpn_boxes, rpn_scores, rpn_anchors = rpn_output

        selected_indices = self.non_max_suppression(rpn_boxes, rpn_scores, threshold)
        selected_rois = [rpn_boxes[i] for i in selected_indices]

        return selected_rois

    def non_max_suppression(self, boxes, scores, threshold):
        if len(boxes) != len(scores):
            raise ValueError("Lists 'boxes' and 'scores' must have the same length.")

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected_indices = []

        while sorted_indices:
            best_idx = sorted_indices[0]
            selected_indices.append(best_idx)
            best_box = boxes[best_idx]
            remaining_indices = []

            for idx in sorted_indices[1:]:
                box = boxes[idx]
                iou = self.calculate_iou(best_box, box)

                if iou < threshold:
                    remaining_indices.append(idx)

            sorted_indices = remaining_indices

        return selected_indices

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection_area / (area1 + area2 - intersection_area)
        return iou

    def call(self, features, rpn_output, training=False):
        rois = self.generate_rois(rpn_output)
        roi_features = self.roi_pooling(rois)
        flattened_features = self.flatten(roi_features)
        fc1_output = self.fc1(flattened_features)
        class_scores = self.cls_head(fc1_output)
        bounding_box_deltas = self.reg_head(fc1_output)
        
        if training:
            return class_scores, bounding_box_deltas
        else:
            return class_scores, bounding_box_deltas, rois
