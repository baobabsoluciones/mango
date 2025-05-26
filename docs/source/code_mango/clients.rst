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

Addresses Data
~~~~~~~~~~~~~~
The Addresses dataset contains entrance points and address information with the following columns:

* ``gml_id`` - Unique GML identifier for the address
* ``localId`` - Local identifier used for linking with buildings
* ``namespace`` - Namespace information for the data source
* ``specification`` - Specification details for the address
* ``method`` - Method used for address determination
* ``default`` - Default value indicator
* ``designator`` - Address number/designator (street number)
* ``type`` - Type of address element
* ``level`` - Level information (floor, building level)
* ``validFrom`` - Date from which the address is valid
* ``beginLifespanVersion`` - Start date of the address version
* ``geometry`` - Point geometry (entrance location)
* ``catastro_municipality_code`` - Municipality code (added by the class)

Buildings Data
~~~~~~~~~~~~~~~
The Buildings dataset contains building footprints and construction information with the following columns:

* ``gml_id`` - Unique GML identifier for the building
* ``lowerCorner`` - Lower corner coordinates of the building bounding box
* ``upperCorner`` - Upper corner coordinates of the building bounding box
* ``beginLifespanVersion`` - Start date of the building version
* ``conditionOfConstruction`` - Construction condition status
* ``beginning`` - Beginning date information
* ``end`` - End date information
* ``endLifespanVersion`` - End date of the building version (if applicable)
* ``informationSystem`` - Information system reference
* ``reference`` - Reference information
* ``localId`` - Local identifier used for linking with addresses
* ``namespace`` - Namespace information for the data source
* ``horizontalGeometryEstimatedAccuracy`` - Estimated accuracy of horizontal geometry
* ``horizontalGeometryEstimatedAccuracy_uom`` - Unit of measure for geometry accuracy
* ``horizontalGeometryReference`` - Reference for horizontal geometry
* ``referenceGeometry`` - Reference geometry information
* ``currentUse`` - Current use classification of the building
* ``numberOfBuildingUnits`` - Number of units in the building
* ``numberOfDwellings`` - Number of dwellings/residential units
* ``numberOfFloorsAboveGround`` - Number of floors above ground level
* ``documentLink`` - Link to related documentation
* ``format`` - Format information
* ``sourceStatus`` - Status of the data source
* ``officialAreaReference`` - Reference to official area measurement
* ``value`` - Area value
* ``value_uom`` - Unit of measure for the area value
* ``geometry`` - Polygon geometry (building footprint)
* ``catastro_municipality_code`` - Municipality code (added by the class)

CadastralParcels Data
~~~~~~~~~~~~~~~~~~~~~
The CadastralParcels dataset contains land parcel information with the following columns:

* ``gml_id`` - Unique GML identifier for the parcel
* ``areaValue`` - Area value of the parcel (numeric)
* ``areaValue_uom`` - Unit of measure for the area value
* ``beginLifespanVersion`` - Start date of the parcel version
* ``endLifespanVersion`` - End date of the parcel version (if applicable)
* ``localId`` - Local identifier for the parcel
* ``namespace`` - Namespace information for the data source
* ``label`` - Label or name for the parcel
* ``nationalCadastralReference`` - National cadastral reference number
* ``pos`` - Position information
* ``geometry`` - Polygon geometry (parcel boundary)
* ``catastro_municipality_code`` - Municipality code (added by the class)

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
   
   # Or link them manually
   addresses = catastro.get_data("28065", "Addresses")
   buildings = catastro.get_data("28065", "Buildings")
   linked_data = catastro.link_entrances_to_buildings(addresses, buildings)

**Data Visualization:**

The cadastral data can be easily visualized using GeoPandas' interactive `explore()` method:

.. code-block:: python

   # Visualize addresses data
   addresses.explore(
       column='type',
       tooltip=['designator', 'type', 'level'],
       popup=True,
       tiles='OpenStreetMap'
   )
   
   # Visualize buildings data
   buildings.explore(
       column='currentUse',
       tooltip=['currentUse', 'numberOfDwellings', 'numberOfFloorsAboveGround'],
       popup=True,
       tiles='OpenStreetMap'
   )
   
   # Visualize cadastral parcels
   parcels.explore(
       column='areaValue',
       tooltip=['nationalCadastralReference', 'areaValue', 'areaValue_uom'],
       popup=True,
       tiles='OpenStreetMap'
   )

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
* Set appropriate ``request_interval`` to respect API rate limits
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
