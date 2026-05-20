from __future__ import annotations

import argparse
import json

from .classifier import CRISPClassifier
from .clustered_classifier import ClusteredCRISPClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="CRISP: Continual Retrieval & Indexing System for Perception")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Create CRISP 1.0 memory bank")
    index_parser.add_argument("--data", required=True)
    index_parser.add_argument("--output", required=True)
    index_parser.add_argument("--encoder", default="resnet50")
    index_parser.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    index_parser.add_argument("--device", default=None)

    predict_parser = subparsers.add_parser("predict", help="Predict using CRISP 1.0")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--memory", required=True)
    predict_parser.add_argument("--encoder", default="resnet50")
    predict_parser.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--voting", choices=["weighted", "majority"], default="weighted")

    index_v2 = subparsers.add_parser("index-v2", help="Create CRISP 2.0 clustered state")
    index_v2.add_argument("--data", required=True)
    index_v2.add_argument("--output", required=True)
    index_v2.add_argument("--router-encoder", default="resnet18")
    index_v2.add_argument("--cluster-method", default="minibatch_kmeans", choices=["kmeans", "minibatch_kmeans"])
    index_v2.add_argument("--n-clusters", type=int, default=3)
    index_v2.add_argument("--cluster-top-m", type=int, default=1)
    index_v2.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    index_v2.add_argument("--device", default=None)
    index_v2.add_argument("--top-k", type=int, default=5)
    index_v2.add_argument("--max-exemplars-per-class", type=int, default=50)
    index_v2.add_argument("--memory-policy", choices=["herding", "none"], default="herding")

    predict_v2 = subparsers.add_parser("predict-v2", help="Predict using CRISP 2.0 clustered state")
    predict_v2.add_argument("--image", required=True)
    predict_v2.add_argument("--state", required=True)
    predict_v2.add_argument("--router-encoder", default="resnet18")
    predict_v2.add_argument("--cluster-method", default="minibatch_kmeans", choices=["kmeans", "minibatch_kmeans"])
    predict_v2.add_argument("--n-clusters", type=int, default=3)
    predict_v2.add_argument("--cluster-top-m", type=int, default=1)
    predict_v2.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    predict_v2.add_argument("--device", default=None)
    predict_v2.add_argument("--top-k", type=int, default=5)
    predict_v2.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    if args.command == "index":
        clf = CRISPClassifier(encoder=args.encoder, retriever=args.retriever, device=args.device)
        clf.add_folder(args.data)
        clf.save(args.output)
        print(f"Saved memory bank to {args.output}")

    elif args.command == "predict":
        clf = CRISPClassifier(
            encoder=args.encoder,
            retriever=args.retriever,
            device=args.device,
            top_k=args.top_k,
            voting=args.voting,
        )
        clf.load(args.memory)
        result = clf.predict(args.image, threshold=args.threshold)
        print(json.dumps(result, indent=2))

    elif args.command == "index-v2":
        clf = ClusteredCRISPClassifier(
            router_encoder=args.router_encoder,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            cluster_top_m=args.cluster_top_m,
            retriever=args.retriever,
            device=args.device,
            top_k=args.top_k,
            max_exemplars_per_class=args.max_exemplars_per_class,
            memory_policy=args.memory_policy,
        )
        clf.add_folder(args.data)
        clf.save(args.output)
        print(f"Saved CRISP 2.0 state to {args.output}")
        print(json.dumps(clf.summary(), indent=2))

    elif args.command == "predict-v2":
        clf = ClusteredCRISPClassifier(
            router_encoder=args.router_encoder,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            cluster_top_m=args.cluster_top_m,
            retriever=args.retriever,
            device=args.device,
            top_k=args.top_k,
        )
        clf.load(args.state)
        result = clf.predict(args.image, threshold=args.threshold)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
