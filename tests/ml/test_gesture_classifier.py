import torch

from ml.models.gesture_classifier import GestureClassifier


def test_gesture_classifier_output_shape():
    model = GestureClassifier(num_classes=4)
    video = torch.rand(2, 16, 3, 224, 224)
    logits = model(video)
    assert logits.shape == (2, 4)


def test_gesture_classifier_requires_multiple_classes():
    try:
        GestureClassifier(num_classes=1)
        assert False, "one-class classifier should be rejected"
    except ValueError:
        pass
