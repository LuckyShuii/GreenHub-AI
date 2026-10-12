"""Catalog mapping for collection materials.

The mapping is intentionally conservative: an item receives a material only
when its name clearly identifies one of the supported collection categories.
"""

from __future__ import annotations

from typing import Final

from .schemas import MaterialMetadata

MATERIAL_CATALOG: Final[dict[str, tuple[str, str, str]]] = {
    "verre": ("VER", "Verre", "verre"),
    "tissus-vetements": ("VET", "Tissus / Vêtements", "tissus-vetements"),
    "plastique": ("PLST", "Plastique", "plastique"),
    "bouchons-plastique": ("BCHP", "Bouchons plastique", "bouchons-plastique"),
    "bouchons-liege": ("BCHL", "Bouchons liège", "bouchons-liege"),
    "papier-emballages": ("PPR", "Papier / Emballages", "papier-emballages"),
    "livres": ("LIV", "Livres", "livres"),
    "CD-DVD": ("CD", "CD / DVD", "CD-DVD"),
    "piles": ("PIL", "Piles", "piles"),
    "chaussures": ("CHS", "Chaussures", "chaussures"),
    "metaux": ("MET", "Métaux", "metaux"),
    "telephones": ("TLF", "Téléphones", "telephones"),
    "dechets-verts": ("GEEN", "Déchets verts", "dechets-verts"),
    "ampoules": ("AMPL", "Ampoules", "ampoules"),
    "cartouches": ("CER", "Cartouches", "cartouches"),
    "materiel-informatique": ("INF", "Matériel informatique", "materiel-informatique"),
    "jouets": ("PLS", "Jouets", "jouets"),
}


def metadata_for_name(name: str) -> MaterialMetadata:
    """Return conservative catalog metadata for an original waste name."""
    normalized = name.casefold()
    matches: list[str] = []

    if "bouchon plastique" in normalized:
        matches = ["bouchons-plastique", "plastique"]
    elif "bouteille" in normalized and "plastique" in normalized:
        matches = ["plastique"]
    elif any(term in normalized for term in ("shampoing", "yaourt", "styromousse", "barquette plastique", "opercule", "pot de fleur plastique", "flacon produit", "bidon plastique", "film plastique", "paille plastique", "gobelet plastique", "emballage chips", "crème cosmétique")):
        matches = ["plastique"]
    elif any(term in normalized for term in ("bouteille en verre", "pot en verre", "bocal confiture")):
        matches = ["verre"]
    elif "vêtement" in normalized:
        matches = ["tissus-vetements"]
    elif "chaussures" in normalized:
        matches = ["chaussures"]
    elif "bouchon liège" in normalized:
        matches = ["bouchons-liege"]
    elif any(term in normalized for term in ("canette", "boîte de conserve", "couvercle métallique", "aérosol vide", "barquette aluminium")):
        matches = ["metaux"]
    elif any(term in normalized for term in ("pile aa", "pile bouton")):
        matches = ["piles"]
    elif "ampoule" in normalized:
        matches = ["ampoules"]
    elif "téléphone portable" in normalized:
        matches = ["telephones"]
    elif "cartouche d'encre" in normalized:
        matches = ["cartouches"]
    elif any(term in normalized for term in ("pc portable", "tablette numérique", "écran tv / moniteur", "disque dur")):
        matches = ["materiel-informatique"]
    elif any(term in normalized for term in ("déchet végétal", "feuilles mortes", "branche de jardin", "gazon tondu")):
        matches = ["dechets-verts"]
    elif any(term in normalized for term in ("papier", "carton", "boîte", "brique de lait", "rouleau carton", "journal", "enveloppe", "annuaire", "post-it")):
        matches = ["papier-emballages"]

    if not matches:
        return MaterialMetadata()
    primary = MATERIAL_CATALOG[matches[0]]
    return MaterialMetadata(
        collection_material_code=primary[0],
        collection_material_name=primary[1],
        collection_materials=matches,
    )
