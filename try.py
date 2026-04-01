"""Load CoDExSmall via PyKEEN and print entity/relation names.

This script only reads names already shipped with the dataset.
It does NOT call Wikidata APIs.
"""

from __future__ import annotations

from typing import Type


def _resolve_codex_small_cls() -> Type:
    """Resolve CoDExSmall class across possible PyKEEN versions."""
    import pykeen.datasets as datasets

    for name in ("CoDExSmall", "CODExSmall"):
        if hasattr(datasets, name):
            return getattr(datasets, name)
    raise ImportError("Cannot find CoDExSmall class in pykeen.datasets")


def main() -> None:
    codex_cls = _resolve_codex_small_cls()
    dataset = codex_cls()  # use PyKEEN default cache path

    tf = dataset.training
    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    print("CoDExSmall loaded.")
    print(f"#entities: {len(id_to_entity)}")
    print(f"#relations: {len(id_to_relation)}")

    print("\nEntity examples:")
    for i in sorted(id_to_entity)[:10]:
        print(f"  entity[{i}] -> {id_to_entity[i]}")

    print("\nRelation examples:")
    for i in sorted(id_to_relation)[:10]:
        print(f"  relation[{i}] -> {id_to_relation[i]}")


if __name__ == "__main__":
    main()

