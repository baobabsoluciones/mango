Clients
--------

AEMET
======
.. autoclass:: mango.clients.aemet.AEMETClient
   :members:
   :undoc-members:
   :show-inheritance:

ARCGIS
======
.. autoclass:: mango.clients.arcgis.ArcGisClient
   :members:
   :undoc-members:
   :show-inheritance:

EmailDownloader
===============
.. autoclass:: mango.clients.email_downloader.EmailDownloader
   :members:
   :undoc-members:
   :show-inheritance:

EmailSender
===========
.. autoclass:: mango.clients.email_sender.EmailSender
   :members:
   :undoc-members:
   :show-inheritance:

GoogleCloudStorage
==================
.. autoclass:: mango.clients.google_cloud_storage.GoogleCloudStorage
   :members:
   :undoc-members:
   :show-inheritance:

RESTClient
==========
.. autoclass:: mango.clients.rest_client.RESTClient
   :members:
   :undoc-members:
   :show-inheritance:

CatastroData
============
The CatastroData class provides access to Spanish cadastral data from the official Catastro API. It supports three main data types:

**Available Data Types:**

1. **Addresses** - Address points and entrance information
2. **Buildings** - Building footprints and construction details  
3. **CadastralParcels** - Land parcel boundaries and cadastral information

**Data Structure and Columns:**

Addresses Dataset
~~~~~~~~~~~~~~~

The Addresses dataset contains entrance points and address information with the following columns:

* ``gml_id``: Unique GML identifier for the address
* ``localId``: Local identifier. Composed of: management code, municipality code, street code, house number plus duplicate, and cadastral reference.
* ``namespace``: Namespace information for the data source. For addresses, this value will be **ES.SDGC.AD**, corresponding to the country, producing organization, and dataset.
* ``specification``: Defines where the address has been georeferenced. In the Cadastre data model, this can take values "Parcel" or, where possible, "Entrance".
* ``method``: Method used for address determination. This will take the value "**fromFeature**".
* ``locator``: Structure that identifies the address by its portal number.
* ``designator``: The portal number, plus the duplicate if it exists.
* ``validFrom``: Date from which the address is valid, which corresponds to when it was registered in the cadastral database.
* ``beginLifespanVersion``: Date from when the address was registered in the cadastral database.
* ``geometry``: Structure containing a point geometry with its coordinates and reference system.
* ``catastro_municipality_code``: Municipality code (added by the class). This information is part of the ``localId``.

Buildings Data
~~~~~~~~~~~~~~~
The Buildings dataset contains building footprints and construction information. There are 3 different subsets available:

* **Buildings**
* **BuildingParts**
* **OtherConstructions**

Buildings
^^^^^^^^^
Building is the main object that defines the building and represents the geometry of the building footprint with a set of attributes defined in an extended 2D schema.

* ``gml_id`` - Unique identifier of the building object, composed of values defined in “inspireID”. Format: ES.SDGC.BU.
* ``lowerCorner`` - Coordinates of the lower-left corner of the building’s bounding rectangle, defined by the coordinate reference system specified in “srsName”.
* ``upperCorner`` - Coordinates of the upper-right corner of the building’s bounding rectangle, defined by the coordinate reference system specified in “srsName”.
* ``beginLifespanVersion`` - Date when the building was added to the cadastral database.
* ``conditionOfConstruction`` - Building conservation state, can take values: ``ruin`` (ruinous), ``declined`` (deficient), or ``functional`` (functional). In multi-unit buildings, the best condition is assigned.
* ``beginning`` - The oldest construction date among units in the building; always referenced to January 1st.
* ``end`` - The most recent construction date among units in the building; always referenced to January 1st.
* ``endLifespanVersion`` - Date the building was removed from the cadastral database. Currently not populated.
* ``informationSystem`` - URL linking to the Spanish Cadastral Electronic Office with detailed cadastral information.
* ``reference`` - Cadastral parcel reference code.
* ``localId`` - First 14 characters of the cadastral reference, part of the unique INSPIRE identifier.
* ``namespace`` - Namespace for INSPIRE datasets, fixed value: ``ES.SDGC.BU`` (Spain, SDGC, Building dataset).
* ``horizontalGeometryEstimatedAccuracy`` - Positional accuracy of the building footprint geometry in meters. Value: 0.1.
* ``horizontalGeometryEstimatedAccuracy_uom`` - Unit of measurement for accuracy. Always in meters.
* ``horizontalGeometryReference`` - Indicates the reference of the geometry. For buildings, the value is always ``footprint``, meaning above-ground footprint.
* ``referenceGeometry`` - Geometry of the building footprint represented as a GML ``gml:Surface`` object, including outer and inner rings with defined coordinate order.
* ``currentUse`` - Dominant use of the building based on the area of the parcels it contains. Possible values:
  - ``1_residential``
  - ``2_agriculture``
  - ``3_industrial``
  - ``4_1_office``
  - ``4_2_retail``
  - ``4_3_publicServices``
* ``numberOfBuildingUnits`` - Total number of units contained within the cadastral parcel of the building.
* ``numberOfDwellings`` - Number of residential units (dwellings) in the cadastral parcel containing the building.
* ``numberOfFloorsAboveGround`` - Number of above-ground floors in the building. This value is specified at the BuildingPart level due to the inability to represent volumetrics at the overall building level in the cadastral model.
* ``documentLink`` - URL linking to a facade photograph of the building. May be unavailable if the image is not in the database.
* ``format`` - Image format of the building photograph. Value: ``jpeg``.
* ``sourceStatus`` - Status of the document source. Value: ``NotOfficial``.
* ``officialAreaReference`` - Indicates the type of surface measured. Always ``grossFloorArea``.
* ``value`` - Official floor area value of the building in square meters.
* ``value_uom`` - Unit of measurement for the floor area value. Typically ``m2``.
* ``geometry`` - Full geometry of the building footprint encoded in GML.
* ``catastro_municipality_code`` - Code identifying the municipality in the Spanish Cadastre system.


BuildingParts
^^^^^^^^^^^^^
BuildingPart refers to each construction within a cadastral parcel that has a homogeneous volume, and can be above or below ground level. It includes attributes related to height.

* ``gml_id`` - Unique identifier for each BuildingPart, composed of the building ID plus a suffix “-PartX”, where X is a sequential number.
* ``beginLifespanVersion`` - Date when the BuildingPart was registered in the cadastral database.
* ``conditionOfConstruction`` - This field is not populated for BuildingParts.
* ``localId`` - Unique local identifier consisting of the first 14 characters of the cadastral reference plus a sequential ``_partX`` suffix.
* ``namespace`` - Namespace for INSPIRE datasets. Fixed value: ``ES.SDGC.BU`` indicating Spain, the SDGC authority, and the building dataset.
* ``horizontalGeometryEstimatedAccuracy`` - Estimated horizontal geometric accuracy in meters. Default value: 0.1.
* ``horizontalGeometryEstimatedAccuracy_uom`` - Unit of measurement for the estimated geometric accuracy. Typically ``m`` (meters).
* ``horizontalGeometryReference`` - Indicates the nature of the geometry. For BuildingParts, always ``footprint``, referring to the above-ground building footprint.
* ``referenceGeometry`` - Geometry of the BuildingPart represented as a GML ``gml:Surface`` object. The geometry includes outer and optional inner rings, following clockwise and counterclockwise vertex orders, respectively.
* ``numberOfFloorsAboveGround`` - Number of above-ground floors for the BuildingPart.
* ``heightBelowGround`` - Estimated height in meters of the underground levels. Calculated based on an estimate of 3 meters per underground floor.
* ``heightBelowGround_uom`` - Unit of measurement for the below-ground height. Typically ``m`` (meters).
* ``numberOfFloorsBelowGround`` - Number of underground floors in the BuildingPart.
* ``geometry`` - Full geometry of the BuildingPart in GML format.
* ``catastro_municipality_code`` - Code representing the municipality in the Spanish Cadastre system.


OtherConstructions
^^^^^^^^^^^^^^^^^^
In this cadastral dataset, we only consider swimming pools that contain the attribute ``OtherConstructionNatureValue`` qualified as ``openAirPool``.

* ``gml_id`` - Unique identifier for each OtherConstruction object, composed of the building ID plus a suffix “-PI.X”, where X is a sequential number.
* ``lowerCorner`` - Coordinates of the lower-left corner of the bounding box that encloses the geometry of the construction.
* ``upperCorner`` - Coordinates of the upper-right corner of the bounding box that encloses the geometry of the construction.
* ``beginLifespanVersion`` - Date when the OtherConstruction object was registered in the cadastral database.
* ``conditionOfConstruction`` - This field is not populated for OtherConstruction elements.
* ``localId`` - Unique local identifier consisting of the first 14 characters of the cadastral reference plus a sequential ``_PI.X`` suffix.
* ``namespace`` - Namespace for INSPIRE datasets. Fixed value: ``ES.SDGC.BU`` indicating Spain, the SDGC authority, and the building dataset.
* ``constructionNature`` - Type of construction. In this cadastral model, the only value used is ``openAirPool`` for representing swimming pools.
* ``geometry`` - Full geometry of the OtherConstruction object in GML format. Defined using a ``gml:Surface`` structure with vertex coordinates in a closed outer ring (clockwise) and possible inner rings (counterclockwise).
* ``catastro_municipality_code`` - Code representing the municipality in the Spanish Cadastre system.


CadastralParcels Data
~~~~~~~~~~~~~~~~~~~~~
The CadastralParcels dataset contains land parcel information. There are 2 different subsets available:

* **CadastralParcels** - Main cadastral parcel information
* **CadastralZoning** - This dataset defines urban blocks (manzanas) in urban areas and polygon zones in rural areas, grouping cadastral parcels for administrative and planning purposes.

CadastralParcels
^^^^^^^^^^^^^^^^^^

* ``gml_id`` - Unique identifier for each CadastralParcel object, composed of the values defined in `inspireId`.
* ``areaValue`` - Surface area of the cadastral parcel expressed in square meters.
* ``areaValue_uom`` - Unit of measurement for `areaValue`. Always expressed in square meters (m²).
* ``beginLifespanVersion`` - Date from which the parcel is considered active in the cadastral database.
* ``endLifespanVersion`` - Date when the parcel was removed from the cadastral database. This value is currently not provided.
* ``localId`` - Local identifier made up of the first 14 characters of the national cadastral reference.
* ``namespace`` - Fixed value: `ES.SDGC.CP`, identifying the dataset as coming from the Spanish Cadastre and referring to cadastral parcels.
* ``label`` - Visible parcel number as shown in cadastral maps. Two digits for urban parcels; up to five digits for rural parcels.
* ``nationalCadastralReference`` - Official cadastral reference code used nationally.
* ``pos`` - Coordinates of the reference point (centroid) of the parcel, typically used to position the parcel label in visualization services.
* ``geometry`` - Full GML geometry of the parcel using a `gml:MultiSurface` structure that includes one or more `gml:Surface` elements, each with rings of coordinates.
* ``catastro_municipality_code`` - Code identifying the municipality as defined in the Spanish Cadastre system.


CadastralZoning
^^^^^^^^^^^^^^^^^^

* ``gml_id`` - Unique identifier of the cadastral zoning feature, composed from the INSPIRE ID values.
* ``beginLifespanVersion`` - Date when the cadastral zoning was added to the cadastral database.
* ``endLifespanVersion`` - Date when the cadastral zoning was removed from the cadastral database (currently not provided).
* ``estimatedAccuracy`` - Estimated geometric accuracy in meters, based on capture scale.
* ``estimatedAccuracy_uom`` - Unit of measure for estimatedAccuracy.
* ``localId`` - Zone code: 12 characters for urban zones (manzanas), 9 characters for rural zones (polígonos).
* ``namespace`` - Namespace identifying country, producer, dataset, and object (ES.SDGC.CP.Z).
* ``label`` - Number identifying the urban block or rural polygon, shown on maps (5 digits urban, up to 3 digits rural).
* ``LocalisedCharacterString`` - Name of hierarchical division, either MANZANA (urban block) or POLIGONO (rural polygon).
* ``nationalCadastalZoningReference`` - National reference code for the zone (12 chars urban, 9 chars rural).
* ``originalMapScaleDenominator`` - Scale denominator of original capture (usually 1000 urban, 2000 or 5000 rural).
* ``pos`` - Coordinates of the centroid point of the zone, used for label positioning.
* ``geometry`` - GML geometry as MultiSurface defining shape with exterior and interior rings.
* ``catastro_municipality_code`` - Code identifying the municipality as defined in the Spanish Cadastre system.


**Example Usage:**

Basic data retrieval for Getafe (municipality code 28065):

.. code-block:: python

   from mango.clients.catastro import CatastroData
   
   # Initialize with caching for better performance
   catastro = CatastroData(cache=True, cache_file_path="catastro_cache.json")
   
   # Get addresses data for Getafe
   addresses = catastro.get_data("28065", "Addresses")
   print(f"Found {len(addresses)} addresses")
   print("Columns:", addresses.columns.tolist())
   
   # Example: Show first few addresses with their designators
   print(addresses[['gml_id', 'designator', 'type', 'level']].head())
   
   # Get buildings data for Getafe
   buildings = catastro.get_data("28065", "Buildings")
   print(f"Found {len(buildings)} buildings")
   
   # Example: Show building information
   print(buildings[['gml_id', 'currentUse', 'numberOfDwellings', 'numberOfFloorsAboveGround']].head())
   
   # Get cadastral parcels for Getafe
   parcels = catastro.get_data("28065", "CadastralParcels")
   print(f"Found {len(parcels)} parcels")
   
   # Example: Show parcel information
   print(parcels[['gml_id', 'nationalCadastralReference', 'areaValue', 'areaValue_uom']].head())

**Linking Addresses to Buildings:**

The class provides functionality to link entrance addresses to their corresponding buildings:

.. code-block:: python

   # Get matched entrances and buildings in one step
   matched_data = catastro.get_matched_entrance_with_buildings("28065")
   
   # Or link them manually if already retrieved
   addresses = catastro.get_data("28065", "Addresses")
   buildings = catastro.get_data("28065", "Buildings")
   linked_data = catastro.link_entrances_to_buildings(addresses, buildings)

**Data Visualization:**

The cadastral data can be easily visualized using GeoPandas' interactive `explore()` method:

.. code-block:: python

   # Visualize addresses data
   addresses.explore()
   
   # Visualize buildings data
   buildings.explore()
   
   # Visualize cadastral parcels
   parcels.explore()

**Example Visualizations for Getafe:**

Addresses Data Visualization:

.. image:: /static/img/addreses_explore.png
   :alt: Interactive map showing address points in Getafe
   :width: 800px
   :align: center

Buildings Data Visualization:

.. image:: /static/img/buildings_explore.png
   :alt: Interactive map showing building footprints in Getafe colored by current use
   :width: 800px
   :align: center

Cadastral Parcels Visualization:

.. image:: /static/img/parcels_explore.png
   :alt: Interactive map showing cadastral parcels in Getafe colored by area value
   :width: 800px
   :align: center

**Performance Tips:**

* Use caching (``cache=True``) to avoid re-downloading the municipality index
* Set appropriate ``request_interval`` to respect API rate limits. The source is not an API per se, but in case of issues, it can be used to avoid overwhelming the server.
* Use ``target_crs`` parameter to ensure consistent coordinate systems
* For large datasets, consider using ``download_all_data()`` for bulk downloads

**Data Quality Notes:**

* Address data includes entrance points that can be linked to buildings via ``localId``
* Building data contains detailed construction information including number of floors and dwellings
* Cadastral parcels provide legal land boundaries and reference numbers
* All datasets include temporal information (``beginLifespanVersion``, ``endLifespanVersion``)
* Geometry is provided in the original CRS (typically EPSG:25830 for Spain) but can be reprojected

.. autoclass:: mango.clients.catastro.CatastroData
   :members:
   :undoc-members:
   :show-inheritance:

INEData
=======

The ``INEData`` class provides access to Spanish census and demographic data from the Instituto Nacional de Estadística (INE). It supports both national-level datasets (CSV) and province-level census tract data (via the INE API).

**Available Data Types**

1. **National Datasets** — National-level population data by census tract.
   These datasets are downloaded from CSV files provided on the INE website, as the API does not support retrieving the full national dataset.

2. **Province Datasets** — Province-specific population data by census tract.
   These datasets are fetched directly from the INE API. The data is equivalent to the national datasets but allows faster access when working with specific provinces.

**Data Structure and Columns**

Both national and provincial datasets share the same structure with the following columns:

* ``ine_province_code``: Code of the province in which the census tract is located.
* ``ine_province_name``: Name of the province.
* ``ine_municipality_code``: Code of the municipality in which the census tract is located.
* ``ine_municipality_name``: Name of the municipality.
* ``ine_census_tract_code``: Code identifying the census tract.
* ``year``: Year the census data was collected.
* ``sex``: Gender of the population group.
* ``<subgroup column>``: This varies depending on the dataset selected via the ``table_code`` (for API) or ``dataset_key`` (for CSV). Available subgroup types include:

  - ``nationality``: Indicates whether individuals are "Spanish" or "Foreign", or specifies their nationality if among the most common foreign ones.
    *(dataset key: ``poblacion_sexo_pais_nacionalidad_principales`` or ``poblacion_sexo_nacionalidad_esp_ext``)*

  - ``birth_country``: Indicates if individuals were born in "Spain", "Foreign", or a specific foreign country.
    *(dataset key: ``poblacion_sexo_pais_nacimiento_esp_ext`` or ``poblacion_sexo_pais_nacimiento_principales``)*

  - ``birth_residence_relation``: Describes the population's relationship between place of birth and residence, e.g.:
    ``"Same municipality"``, ``"Different municipality, same province"``, ``"Same Autonomous Community"``, ``"Different Autonomous Community"``, ``"Abroad"``.
    *(dataset key: ``poblacion_sexo_relacion_nacimiento_residencia``)*

  - ``age_group``: Grouped age brackets, such as "0–10", "11–20", etc.
    *(dataset key: ``poblacion_sexo_edad_quinquenales``)*

* ``population_count``: Total number of individuals in the group defined by the above attributes.

**Example Usage:**

Basic data retrieval for the province of Madrid grouped by sex and age group:

.. code-block:: python

   from mango.clients.ine import INEData

   # Initialize with caching for better performance
   ine_data = INEData(table_codes_json_path = "ine_table_codes.json")

   # List the table codes to find the one for Madrid
   ine_data.list_table_codes()

   # Get province dataset for Madrid province
   madrid_data = ine_data.get_national_data("69213")

   # Show first few records with population counts
   print(madrid_data.head())


National dataset for Spain grouped by sex and nationality (Spanish or Foreign):

.. code-block:: python

   # Get national dataset for Spain
   national_data = ine_data.get_national_data("poblacion_sexo_pais_nacimiento_esp_ext")

   # Show first few records with population counts
    print(national_data.head())


Additionally, you can add the geometries of the census tracts to any dataset using the enrich_with_geometries() method.

.. code-block:: python

   # Enrich Madrid data with geometries
   madrid_data_with_geometries = ine_data.enrich_with_geometries(madrid_data)

The dataset returned is then in GeoDataFrame format, allowing for easy visualization and spatial analysis.

.. automodule:: mango.clients.ine
   :members:
   :undoc-members:
   :show-inheritance:

CatastroINEMapper
=================
.. automodule:: mango.clients.ine_catastro_mapper
   :members:
   :show-inheritance:

MITMA
=====
.. automodule:: mango.clients.mitma
   :members:
   :undoc-members:
   :show-inheritance:

MitmaINEMapper
==============
.. automodule:: mango.clients.mitma_ine_mapper
   :members:
   :undoc-members:
   :show-inheritance:

