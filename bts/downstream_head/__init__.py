from .semantic_head import SemanticHead


def make_downstream_head(conf):
    head_type = conf.get("type", None)
    if head_type == "segmentation":
        return SemanticHead.from_conf(conf)
    else:
        raise NotImplementedError(f"Downstream head type '{head_type}' is not implemented.")
