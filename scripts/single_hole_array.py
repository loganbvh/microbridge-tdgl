import dataclasses
import multiprocessing as mp
import os
from functools import partial

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
import tdgl
from tdgl.geometry import box, circle


def make_device(
    *,
    width: float = 500.0,
    height: float = 500.0,
    hole_radius: float = 100.0,
    london_lambda: float = 200.0,
    xi: float = 10.0,
    thickness: float = 5.0,
    max_edge_length: float = 5.0,
) -> tdgl.Device:
    layer = tdgl.Layer(
        london_lambda=london_lambda, coherence_length=xi, thickness=thickness
    )
    perimeter = 2 * (width + height)
    boundary_points = int(3 * perimeter / max_edge_length)
    film = tdgl.Polygon("film", points=box(width, height)).resample(boundary_points)
    source = tdgl.Polygon("source", points=box(width / 100, height)).translate(
        dx=-width / 2
    )
    drain = source.scale(xfact=-1).set_name("drain")

    holes = None
    if hole_radius:
        hole_points = int(2 * np.pi * hole_radius)
        hole = tdgl.Polygon("hole", points=circle(hole_radius, points=int(hole_points)))
        holes = [hole]

    device = tdgl.Device(
        "bridge",
        layer=layer,
        film=film,
        holes=holes,
        terminals=[source, drain],
        probe_points=[(-0.8 * width / 2, 0), (+0.8 * width / 2, 0)],
        length_units="nm",
    )

    device.make_mesh(max_edge_length=max_edge_length)

    print(device)
    print(device.mesh_stats_dict())

    return device


def get_dynamics(current, *, device, options, applied_field):
    solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=applied_field,
        terminal_currents=dict(source=current, drain=-current),
    )
    return solution.dynamics


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--width", type=float, default=500, help="Device width in nm")
    parser.add_argument("--height", type=float, default=500, help="Device height in nm")
    parser.add_argument("--radius", type=float, default=100, help="Hole radius in nm")
    parser.add_argument(
        "--lambda_", type=float, default=200, help="London penetration depth in nm"
    )
    parser.add_argument("--xi", type=float, default=10, help="Coherence length in nm")
    parser.add_argument("--d", type=float, default=5, help="Film thickness in nm")
    parser.add_argument(
        "--max-edge-length", type=float, default=None, help="Max edge length in nm"
    )
    parser.add_argument(
        "--currents",
        type=float,
        nargs=3,
        help="Current (start, stop, npoints) in uA",
    )
    parser.add_argument("--index", type=int, help="Index into currents.")
    parser.add_argument(
        "--field",
        type=float,
        help="Applied field in mT",
    )
    parser.add_argument(
        "--solve-time", type=float, default=700, help="Total solve time"
    )
    parser.add_argument(
        "--eval-time", type=float, default=200, help="Voltage evaluation time"
    )
    parser.add_argument("--output", type=str, help="Path to output HDF5 file.")

    args = parser.parse_args()

    start, stop, npoints = args.currents
    currents = np.linspace(start, stop, int(npoints))
    index = int(args.index)
    current = currents[index]

    field = float(args.field)

    max_edge_length = args.max_edge_length
    if max_edge_length is None:
        max_edge_length = args.xi / 2

    device_kwargs = dict(
        width=args.width,
        height=args.height,
        hole_radius=args.radius,
        london_lambda=args.lambda_,
        xi=args.xi,
        thickness=args.d,
        max_edge_length=max_edge_length,
    )

    device = make_device(**device_kwargs)

    options = tdgl.SolverOptions(
        solve_time=args.solve_time,
        save_every=500,
        current_units="uA",
        field_units="mT",
    )

    with h5py.File(args.output, "x") as h5file:
        if index == 0:
            device.to_hdf5(h5file.create_group("device"))
        h5file["field"] = field
        h5file["currents"] = currents
        h5file["index"] = index
        h5file["current"] = current
        for k, v in device_kwargs.items():
            h5file.attrs[k] = v
        opt_grp = h5file.create_group("options")
        for k, v in dataclasses.asdict(options).items():
            if v is not None:
                opt_grp.attrs["k"] = v

        dynamics = get_dynamics(
            current, device=device, options=options, applied_field=field
        )
        voltage = dynamics.mean_voltage(tmin=(args.solve_time - args.eval_time))
        h5file["voltage"] = voltage
        dynamics.to_hdf5(h5file.create_group("dynamics"))
        print(f"{index}: I = {current:.5f} uA, V = {voltage:.5f} V0")
