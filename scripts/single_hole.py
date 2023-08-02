import dataclasses
import multiprocessing as mp
import os
from functools import partial

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import joblib
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
    film = tdgl.Polygon("film", points=box(width, height, points=2001)).resample(1001)
    source = tdgl.Polygon("source", points=box(width / 100, height)).translate(
        dx=-width / 2
    )
    drain = source.scale(xfact=-1).set_name("drain")

    holes = None
    if hole_radius:
        holes = [tdgl.Polygon("hole", points=circle(hole_radius, points=int(4 * hole_radius)))]

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
        "--ncpus", type=int, default=None, help="Number of processes to use"
    )
    parser.add_argument(
        "--currents",
        type=float,
        nargs=3,
        help="Current (start, stop, num_points) in uA",
    )
    parser.add_argument(
        "--fields",
        type=float,
        nargs=3,
        help="Applied field (start, stop, num_points) in mT",
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

    start, stop, npoints = args.fields
    fields = np.linspace(start, stop, int(npoints))

    avail_cpus = joblib.cpu_count(only_physical_cores=True)
    ncpus = args.ncpus
    if ncpus is None:
        ncpus = avail_cpus
    ncpus = min(ncpus, avail_cpus)

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

    with h5py.File(args.output, "x", track_order=True) as h5file:
        device.to_hdf5(h5file.create_group("device"))
        for k, v in device_kwargs.items():
            h5file.attrs[k] = v
        h5file.attrs["ncpus"] = ncpus
        opt_grp = h5file.create_group("options")
        for k, v in dataclasses.asdict(options).items():
            if v is not None:
                opt_grp.attrs["k"] = v

    for i, field in enumerate(fields):
        print(f"{i}: {field:.5f} mT")
        with mp.Pool(processes=ncpus) as pool:
            func = partial(
                get_dynamics, device=device, options=options, applied_field=field
            )
            results = pool.map(func, currents)

        with h5py.File(args.output, "a") as f:
            grp = f.require_group(str(i))
            grp.attrs["field"] = field
            voltages = []
            for j, (result, current) in enumerate(zip(results, currents)):
                vmean = result.mean_voltage(tmin=(args.solve_time - args.eval_time))
                voltages.append(vmean)
                subgrp = grp.create_group(str(j))
                subgrp.attrs["current"] = current
                result.to_hdf5(subgrp)
            voltages = np.array(voltages)
            grp["voltage"] = voltages
            print(currents)
            print(voltages)
