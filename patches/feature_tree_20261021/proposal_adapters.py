"""Select the proposal runtime without changing the saved decoder."""


def build_proposal(adapter, verify=False):
    if adapter.name in {"pf_mlp", "medusa_lowrank"}:
        from mlp_medusa_graph import build_proposal as build_dense
        return build_dense(adapter, verify=verify)
    if adapter.variant == "eagle":
        from eagle_graph import build_proposal as build_eagle
        return build_eagle(adapter, verify=verify)
    raise ValueError(f"No proposal graph for {adapter.name}")
