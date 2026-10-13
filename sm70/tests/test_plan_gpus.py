"""scripts/plan_gpus.py on recorded topologies (CPU only, standard library).

Fixtures in topologies/: host4_sxm2_32g is captured from the reference host (UUIDs replaced); the others
are synthetic: dgx1_* is the DGX-1 (V100) hybrid cube-mesh as published by NVIDIA, station4_16g a
4-GPU board with every pair on NVLink, pcie8_* two sockets of PCIe cards without NVLink.
"""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plan_gpus", os.path.join(HERE, "..", "..", "scripts", "plan_gpus.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def load(name):
    d = os.path.join(HERE, "topologies")
    gpus = P.parse_query(open(os.path.join(d, name + ".query.txt")).read())
    links, aff = P.parse_topo(open(os.path.join(d, name + ".topo.txt")).read())
    return gpus, links, aff


def groups(p):
    return [tuple(r["cards"]) for r in p["replicas"]]


def test_parse_real_topo_with_ansi_header():
    gpus, links, aff = load("host4_sxm2_32g")
    assert len(gpus) == 4 and gpus[0]["mem_mib"] == 32768
    assert links[(2, 3)] == "NV2" and links[(0, 1)] == "PHB" and links[(1, 2)] == "NODE"
    assert links[(3, 2)] == "NV2" and aff[0] == "0-23"


def test_host_default_two_pairs():
    p = P.plan(*load("host4_sxm2_32g"))
    assert p["profile"] == "tp2-32g" and groups(p) == [(0, 1), (2, 3)]
    ex = {tuple(r["cards"]): r["env"]["FA2SM70_EXCH12"] for r in p["replicas"]}
    assert ex == {(0, 1): "1", (2, 3): "0"}            # PCIe pair -> int12, NVLink pair -> fp16
    assert all(r["validated"] for r in p["replicas"])


def test_host_allowed_subset_by_index_and_uuid():
    g = load("host4_sxm2_32g")
    assert groups(P.plan(*g, allowed="1,2")) == [(1, 2)]
    u = "GPU-00000000-0000-0000-0000-000000000003,GPU-00000000-0000-0000-0000-000000000002"
    p = P.plan(*g, allowed=u)
    assert groups(p) == [(2, 3)] and p["replicas"][0]["uuids"][0].endswith("02")


def test_host_tp4_partial_nvlink():
    p = P.plan(*load("host4_sxm2_32g"), tp=4)
    r = p["replicas"][0]
    assert groups(p) == [(0, 1, 2, 3)] and r["env"]["FA2SM70_EXCH12"] == "0"
    assert r["exchange"].startswith("partial NVLink") and not r["validated"]


def test_odd_count_leaves_a_card_unused():
    p = P.plan(*load("host4_sxm2_32g"), allowed="0,2,3")
    assert groups(p) == [(2, 3)] and p["unused"] == [0] and p["warnings"]


def test_dgx1_32g_four_nvlink2_pairs_on_one_socket_each():
    p = P.plan(*load("dgx1_8x_32g"))
    assert p["tp"] == 2 and len(p["replicas"]) == 4
    for r in p["replicas"]:
        assert list(r["links"].values()) == ["NV2"] and r["same_numa"] and r["env"]["FA2SM70_EXCH12"] == "0"
    assert sorted(groups(p)) == [(0, 3), (1, 2), (4, 7), (5, 6)]


def test_dgx1_16g_two_fully_linked_quads():
    p = P.plan(*load("dgx1_8x_16g"))
    assert p["profile"] == "tp4-16g" and groups(p) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    for r in p["replicas"]:
        assert r["exchange"].startswith("all pairs NVLink") and r["same_numa"] and r["cpuset"]


def test_dgx1_single_tp8():
    p = P.plan(*load("dgx1_8x_16g"), single=True)
    assert p["tp"] == 8 and groups(p) == [tuple(range(8))]
    assert p["replicas"][0]["exchange"].startswith("partial NVLink")


def test_station_16g_needs_tp4():
    g = load("station4_16g")
    assert groups(P.plan(*g)) == [(0, 1, 2, 3)]
    with pytest.raises(SystemExit):
        P.plan(*g, tp=2)                                 # 27B int8 does not fit two 16 GB cards


def test_pcie_two_sockets_never_spans_sockets():
    p = P.plan(*load("pcie8_32g_2socket"))
    assert sorted(groups(p)) == [(0, 1), (2, 3), (4, 5), (6, 7)]  # PIX pairs behind one switch
    assert all(r["env"]["FA2SM70_EXCH12"] == "1" for r in p["replicas"])
    p = P.plan(*load("pcie8_16g_2socket"))
    assert groups(p) == [(0, 1, 2, 3), (4, 5, 6, 7)] and all(r["same_numa"] for r in p["replicas"])


def test_mixed_card_sizes_sized_for_the_smallest():
    gpus, links, aff = load("dgx1_8x_32g")
    gpus[5]["mem_mib"] = 16160
    p = P.plan(gpus, links, aff)
    assert p["tp"] == 4 and any("mixed" in w for w in p["warnings"])


def test_bad_requests_refused():
    g = load("host4_sxm2_32g")
    for kw in ({"allowed": "0,0"}, {"allowed": "7"}, {"allowed": "GPU-nope"}, {"single": True, "allowed": "0,1,2"}):
        with pytest.raises(SystemExit):
            P.plan(*g, **kw)


def test_compose_and_nginx_render():
    p = P.plan(*load("dgx1_8x_32g"))
    y = P.render_compose(p, 8000, "cdi")
    for r in p["replicas"]:
        for u in r["uuids"]:
            assert f"nvidia.com/gpu={u}" in y
    assert y.count("image: ${SM70_IMAGE:-vllm-sm70:latest}") == 4 and '"127.0.0.1:8004:8000"' in y
    assert "driver: nvidia" in P.render_compose(p, 8000, "nvidia")
    assert P.render_nginx(p).count("server sm70-") == 4
