#!/usr/bin/env python3
"""Rebin CaloChallenge dataset 3 from 45x50x18 to 45x25x9.

The source CCD3 HDF5 stores showers flattened in the evaluator order
`(z, phi, r) = (45, 50, 18)`. For many modeling paths it is more convenient to
think in the internal tensor order `(r, phi, z) = (18, 50, 45)`.

This script rebins by merging each adjacent pair of phi bins and each adjacent
pair of r bins with simple sums, leaving z/layer unchanged. The logical output
shape is therefore:

- evaluator order: `(45, 25, 9)`
- internal order: `(9, 25, 45)`

By default the output HDF5 keeps `showers` flattened in evaluator order so it
can be paired directly with the generated XML file. The internal-order shape is
recorded in HDF5 attributes for bookkeeping.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

INPUT_SHAPE_Z_PHI_R = (45, 50, 18)
OUTPUT_SHAPE_Z_PHI_R = (45, 25, 9)
OUTPUT_SHAPE_R_PHI_Z = (9, 25, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebin CCD3 dataset3 by 2x2 merging in (phi, r)."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Input dataset_3 HDF5 file with flat 40500-voxel showers.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        required=True,
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--input-xml",
        type=Path,
        default=Path("cc_metrics/binning_dataset_3.xml"),
        help="Input CCD3 XML used as the source geometry.",
    )
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=None,
        help="Output XML path. Defaults to <output-h5 stem>.xml.",
    )
    parser.add_argument(
        "--chunk-showers",
        type=int,
        default=512,
        help="Number of showers to process per chunk.",
    )
    parser.add_argument(
        "--output-layout",
        choices=("flat-z-phi-r", "tensor-r-phi-z"),
        default="flat-z-phi-r",
        help=(
            "How to store the output `showers` dataset. "
            "`flat-z-phi-r` keeps evaluator-compatible flat rows of length 10125. "
            "`tensor-r-phi-z` writes a 4D dataset with shape (N, 9, 25, 45)."
        ),
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=4,
        help="Gzip compression level for the output HDF5.",
    )
    return parser.parse_args()


def find_energy_key(handle: h5py.File) -> str:
    if "incident_energies" in handle:
        return "incident_energies"
    if "incident_energy" in handle:
        return "incident_energy"
    raise KeyError("Could not find `incident_energies` or `incident_energy` in input file.")


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def rebin_showers_z_phi_r(flat_chunk: np.ndarray) -> np.ndarray:
    """Merge adjacent phi/r bins while keeping the evaluator `(z, phi, r)` order."""
    batch = flat_chunk.shape[0]
    showers = flat_chunk.reshape(batch, *INPUT_SHAPE_Z_PHI_R)
    rebinned = showers.reshape(batch, 45, 25, 2, 9, 2).sum(axis=(3, 5), dtype=np.float64)
    return rebinned.astype(np.float32, copy=False)


def format_showers_for_output(showers_z_phi_r: np.ndarray, layout: str) -> np.ndarray:
    if layout == "flat-z-phi-r":
        return showers_z_phi_r.reshape(showers_z_phi_r.shape[0], -1)
    if layout == "tensor-r-phi-z":
        return showers_z_phi_r.transpose(0, 3, 2, 1)
    raise ValueError(f"Unsupported output layout: {layout}")


def create_showers_dataset(
    dst: h5py.File,
    src_showers: h5py.Dataset,
    total_events: int,
    layout: str,
    compression: int,
):
    if layout == "flat-z-phi-r":
        shape = (total_events, int(np.prod(OUTPUT_SHAPE_Z_PHI_R)))
        chunks = src_showers.chunks
        if chunks is not None:
            chunks = (min(chunks[0], total_events), shape[1])
    else:
        shape = (total_events, *OUTPUT_SHAPE_R_PHI_Z)
        chunks = src_showers.chunks
        if chunks is not None:
            chunks = (min(chunks[0], total_events), *OUTPUT_SHAPE_R_PHI_Z)

    dataset = dst.create_dataset(
        "showers",
        shape=shape,
        dtype=np.float32,
        chunks=chunks,
        compression="gzip",
        compression_opts=compression,
    )
    copy_attrs(src_showers, dataset)
    dataset.attrs["source_shape_z_phi_r"] = INPUT_SHAPE_Z_PHI_R
    dataset.attrs["rebinned_shape_z_phi_r"] = OUTPUT_SHAPE_Z_PHI_R
    dataset.attrs["rebinned_shape_r_phi_z"] = OUTPUT_SHAPE_R_PHI_Z
    dataset.attrs["rebin_rule"] = "sum adjacent phi pairs and adjacent r pairs"
    dataset.attrs["storage_layout"] = layout
    return dataset


def create_like_dataset(dst: h5py.File, name: str, src_dataset: h5py.Dataset):
    dataset = dst.create_dataset(
        name,
        shape=src_dataset.shape,
        dtype=src_dataset.dtype,
        chunks=src_dataset.chunks,
        compression=src_dataset.compression,
        compression_opts=src_dataset.compression_opts,
        shuffle=src_dataset.shuffle,
        fletcher32=src_dataset.fletcher32,
        scaleoffset=src_dataset.scaleoffset,
    )
    copy_attrs(src_dataset, dataset)
    return dataset


def write_rebinned_h5(
    input_h5: Path,
    output_h5: Path,
    output_layout: str,
    chunk_showers: int,
    compression: int,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["source_file"] = str(input_h5.resolve())
        dst.attrs["source_shape_z_phi_r"] = INPUT_SHAPE_Z_PHI_R
        dst.attrs["rebinned_shape_z_phi_r"] = OUTPUT_SHAPE_Z_PHI_R
        dst.attrs["rebinned_shape_r_phi_z"] = OUTPUT_SHAPE_R_PHI_Z
        dst.attrs["showers_storage_layout"] = output_layout

        src_showers = src["showers"]
        total_events, input_voxels = src_showers.shape
        expected_voxels = int(np.prod(INPUT_SHAPE_Z_PHI_R))
        if int(input_voxels) != expected_voxels:
            raise ValueError(
                f"Expected {expected_voxels} voxels for CCD3, got {input_voxels} from {input_h5}."
            )

        showers_out = create_showers_dataset(
            dst,
            src_showers=src_showers,
            total_events=int(total_events),
            layout=output_layout,
            compression=compression,
        )

        for key in src.keys():
            if key == "showers":
                continue
            dst_dataset = create_like_dataset(dst, key, src[key])
            for start in range(0, int(total_events), chunk_showers):
                stop = min(start + chunk_showers, int(total_events))
                dst_dataset[start:stop] = src[key][start:stop]

        max_energy_diff = 0.0
        for start in range(0, int(total_events), chunk_showers):
            stop = min(start + chunk_showers, int(total_events))
            chunk = src_showers[start:stop].astype(np.float32, copy=False)
            rebinned_z_phi_r = rebin_showers_z_phi_r(chunk)
            formatted = format_showers_for_output(rebinned_z_phi_r, output_layout)
            showers_out[start:stop] = formatted

            original_sum = chunk.sum(axis=1, dtype=np.float64)
            rebinned_sum = rebinned_z_phi_r.reshape(rebinned_z_phi_r.shape[0], -1).sum(
                axis=1, dtype=np.float64
            )
            max_energy_diff = max(max_energy_diff, float(np.max(np.abs(original_sum - rebinned_sum))))

        dst.attrs["max_abs_event_energy_diff_after_rebin"] = max_energy_diff


def build_rebinned_xml(input_xml: Path, output_xml: Path) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()

    for particle in root:
        for layer in particle:
            edges = [float(value) for value in layer.attrib["r_edges"].split(",")]
            if (len(edges) - 1) % 2 != 0:
                raise ValueError(
                    f"Layer {layer.attrib.get('id')} has {len(edges) - 1} r bins; expected an even count."
                )
            new_edges = edges[::2]
            if new_edges[-1] != edges[-1]:
                new_edges.append(edges[-1])

            alpha_bins = int(layer.attrib["n_bin_alpha"])
            if alpha_bins % 2 != 0:
                raise ValueError(
                    f"Layer {layer.attrib.get('id')} has {alpha_bins} alpha bins; expected an even count."
                )

            layer.set("r_edges", ",".join(f"{edge:g}" for edge in new_edges))
            layer.set("n_bin_alpha", str(alpha_bins // 2))

    if hasattr(ET, "indent"):
        ET.indent(tree, space="   ")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def main() -> None:
    args = parse_args()
    output_xml = args.output_xml
    if output_xml is None:
        output_xml = args.output_h5.with_suffix(".xml")

    build_rebinned_xml(args.input_xml, output_xml)
    write_rebinned_h5(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        output_layout=args.output_layout,
        chunk_showers=max(1, args.chunk_showers),
        compression=args.compression,
    )

    print(f"rebinned_h5: {args.output_h5}")
    print(f"rebinned_xml: {output_xml}")
    print(f"logical_shape_z_phi_r: {OUTPUT_SHAPE_Z_PHI_R}")
    print(f"logical_shape_r_phi_z: {OUTPUT_SHAPE_R_PHI_Z}")
    print(f"showers_storage_layout: {args.output_layout}")


if __name__ == "__main__":
    main()
