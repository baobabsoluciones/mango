"""
Sample file generators for CatastroData tests.

This module provides functions to create mock ZIP and GML files
for testing different scenarios without requiring real files.
"""

import zipfile
from io import BytesIO
from typing import Dict, List, Optional


def create_mock_zip_with_gml(
    gml_content: str = "<gml>test content</gml>", filename: str = "test.building.gml"
) -> bytes:
    """
    Create a valid ZIP file containing a GML file.

    :param gml_content: Content to put in the GML file.
    :param filename: Name of the GML file inside the ZIP.
    :return: ZIP file content as bytes.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr(filename, gml_content.encode("utf-8"))
    return zip_buffer.getvalue()


def create_mock_zip_with_multiple_files(files: Dict[str, str]) -> bytes:
    """
    Create a ZIP file with multiple files.

    :param files: Dictionary mapping filename to content.
    :return: ZIP file content as bytes.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content.encode("utf-8"))
    return zip_buffer.getvalue()


def create_mock_zip_without_gml() -> bytes:
    """
    Create a ZIP file without GML files (for error testing).

    :return: ZIP file content as bytes.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("readme.txt", b"No GML files here")
        zf.writestr("data.csv", b"col1,col2\nval1,val2")
    return zip_buffer.getvalue()


def create_invalid_zip() -> bytes:
    """
    Create invalid ZIP data for error testing.

    :return: Invalid ZIP content as bytes.
    """
    return b"This is not a valid ZIP file content"


def create_mock_building_gml() -> str:
    """
    Create a realistic mock GML content for buildings.

    :return: GML content as string.
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2">
    <gml:featureMember>
        <bu:Building xmlns:bu="http://inspire.ec.europa.eu/schemas/bu-core3d/4.0">
            <bu:inspireId>
                <base:Identifier xmlns:base="http://inspire.ec.europa.eu/schemas/base/3.3">
                    <base:localId>building1</base:localId>
                    <base:namespace>ES.SDGC.BU</base:namespace>
                </base:Identifier>
            </bu:inspireId>
            <bu:geometry>
                <gml:MultiSurface>
                    <gml:surfaceMember>
                        <gml:Polygon>
                            <gml:exterior>
                                <gml:LinearRing>
                                    <gml:posList>0 0 1 0 1 1 0 1 0 0</gml:posList>
                                </gml:LinearRing>
                            </gml:exterior>
                        </gml:Polygon>
                    </gml:surfaceMember>
                </gml:MultiSurface>
            </bu:geometry>
        </bu:Building>
    </gml:featureMember>
</gml:FeatureCollection>"""


def create_mock_address_gml() -> str:
    """
    Create a realistic mock GML content for addresses.

    :return: GML content as string.
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2">
    <gml:featureMember>
        <ad:Address xmlns:ad="http://inspire.ec.europa.eu/schemas/ad/4.0">
            <ad:inspireId>
                <base:Identifier xmlns:base="http://inspire.ec.europa.eu/schemas/base/3.3">
                    <base:localId>address1.building1</base:localId>
                    <base:namespace>ES.SDGC.AD</base:namespace>
                </base:Identifier>
            </ad:inspireId>
            <ad:position>
                <ad:GeographicPosition>
                    <ad:geometry>
                        <gml:Point>
                            <gml:pos>0.5 0.5</gml:pos>
                        </gml:Point>
                    </ad:geometry>
                </ad:GeographicPosition>
            </ad:position>
        </ad:Address>
    </gml:featureMember>
</gml:FeatureCollection>"""


def create_mock_cadastral_parcel_gml() -> str:
    """
    Create a realistic mock GML content for cadastral parcels.

    :return: GML content as string.
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2">
    <gml:featureMember>
        <cp:CadastralParcel xmlns:cp="http://inspire.ec.europa.eu/schemas/cp/4.0">
            <cp:inspireId>
                <base:Identifier xmlns:base="http://inspire.ec.europa.eu/schemas/base/3.3">
                    <base:localId>parcel1</base:localId>
                    <base:namespace>ES.SDGC.CP</base:namespace>
                </base:Identifier>
            </cp:inspireId>
            <cp:geometry>
                <gml:MultiSurface>
                    <gml:surfaceMember>
                        <gml:Polygon>
                            <gml:exterior>
                                <gml:LinearRing>
                                    <gml:posList>0 0 2 0 2 2 0 2 0 0</gml:posList>
                                </gml:LinearRing>
                            </gml:exterior>
                        </gml:Polygon>
                    </gml:surfaceMember>
                </gml:MultiSurface>
            </cp:geometry>
        </cp:CadastralParcel>
    </gml:featureMember>
</gml:FeatureCollection>"""


# Predefined ZIP files for common test scenarios
MOCK_ZIP_FILES = {
    "buildings": create_mock_zip_with_gml(
        create_mock_building_gml(), "test.building.gml"
    ),
    "addresses": create_mock_zip_with_gml(
        create_mock_address_gml(), "test.address.gml"
    ),
    "cadastral_parcels": create_mock_zip_with_gml(
        create_mock_cadastral_parcel_gml(), "test.cadastralparcel.gml"
    ),
    "empty": create_mock_zip_without_gml(),
    "invalid": create_invalid_zip(),
}
