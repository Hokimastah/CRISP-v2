from crisp import ClusteredCRISPClassifier


def main():
    clf = ClusteredCRISPClassifier(
        router_encoder="resnet18",
        cluster_method="minibatch_kmeans",
        n_clusters=3,
        cluster_top_m=1,
        specialist_backbones={0: "resnet18", 1: "resnet50", 2: "resnet50"},
        retriever="numpy",
        voting="weighted",
        max_exemplars_per_class=50,
        device="cpu",
    )

    clf.add_folder("dataset")
    clf.save("crisp_v2_state.pkl")

    result = clf.predict("test_image.jpg")
    print(result["predicted_label"])
    print(result["scores"])
    print(clf.summary())


if __name__ == "__main__":
    main()
