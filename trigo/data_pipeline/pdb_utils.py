from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def _resname_to_one_letter(resname: str) -> str:
    return THREE_TO_ONE.get(resname.upper(), "X")


def parse_pdb_ca(
    pdb_path: str | Path,
    chain_id: Optional[str] = "A",
) -> tuple[str, Tensor, List[str]]:
    """
    Parse a PDB file and extract C-alpha coordinates.

    Returns:
    - sequence (str): residue sequence inferred from PDB
    - coords (Tensor): [L, 3]
    - residue_keys (List[str]): unique residue identifiers
    """
    pdb_path = Path(pdb_path)
    sequence: List[str] = []
    coords: List[List[float]] = []
    residue_keys: List[str] = []
    seen = set()

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            current_chain = line[21].strip() or "A"
            if chain_id is not None and current_chain != chain_id:
                continue

            alt_loc = line[16].strip()
            if alt_loc not in {"", "A"}:
                continue

            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = f"{current_chain}:{resseq}:{icode}"
            if residue_key in seen:
                continue
            seen.add(residue_key)

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            residue_keys.append(residue_key)
            sequence.append(_resname_to_one_letter(resname))
            coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No C-alpha atoms were found in PDB: {pdb_path}")

    return "".join(sequence), torch.tensor(coords, dtype=torch.float32), residue_keys


def build_contact_map(
    ca_coords: Tensor,
    distance_threshold: float = 8.0,
    include_self: bool = True,
) -> Tensor:
    """
    Build binary residue contact map from C-alpha coordinates.
    """
    dists = torch.cdist(ca_coords, ca_coords, p=2.0)
    contact_map = (dists <= distance_threshold).to(torch.float32)
    if not include_self:
        contact_map.fill_diagonal_(0.0)
    return contact_map


def truncate_residue_level_features(
    sequence: str,
    ca_coords: Tensor,
    max_residues: int,
) -> tuple[str, Tensor]:
    if max_residues <= 0:
        return sequence, ca_coords
    if len(sequence) <= max_residues:
        return sequence, ca_coords
    return sequence[:max_residues], ca_coords[:max_residues]

