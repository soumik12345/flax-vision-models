import os
import wget
import json
import wandb


CATEGORIES_ARTIFACT_ADDRESS = (
    "geekyrakshit/flax-vision-models/imagenet-simple-labels:v0"
)
CATEGORIES_FILE_ADDRESS = "imagenet-simple-labels.json"

CATEGORIES_FILE_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def decode_probabilities_imagenet(topk_classes, topk_probabilities):
    imagenet_labels_file = None
    if wandb.run is not None:
        artifact = wandb.use_artifact(CATEGORIES_ARTIFACT_ADDRESS)
        artifact_dir = artifact.download()
        imagenet_labels_file = os.path.join(artifact_dir, CATEGORIES_FILE_ADDRESS)
    else:
        imagenet_labels_file = wget.download(CATEGORIES_FILE_URL)
    categories = json.load(open(imagenet_labels_file))
    topk_labels = [categories[topk_classes[i]] for i in range(topk_classes.shape[0])]
    topk_probabilities = [topk_probabilities[i] for i in range(topk_classes.shape[0])]
    if wandb.run is not None:
        wandb.log(
            {
                "Prediction": wandb.plot.bar(
                    wandb.Table(
                        data=[
                            [label, val]
                            for (label, val) in zip(topk_labels, topk_probabilities)
                        ],
                        columns=["Predicted Label", "Probability"],
                    ),
                    "Predicted Label",
                    "Probability",
                    title="Prediction",
                )
            }
        )
    return topk_labels, topk_probabilities
