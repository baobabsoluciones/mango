import pandas as pd
from ine_catastro_mapper import CatastroINEMapper # You might need to adjust import paths
from sklearn.cluster import KMeans
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def merge_catastro_census(cadastral_addresses_with_buildings: gpd.GeoDataFrame, census_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merges cadastral address and building data with census data based on spatial relationships
    and municipality code matching, correcting mismatches by finding the nearest census unit.

    :param cadastral_addresses_with_buildings: GeoDataFrame containing cadastral addresses and buildings.
    :type cadastral_addresses_with_buildings: gpd.GeoDataFrame
    :param census_data: GeoDataFrame containing census data.
    :type census_data: gpd.GeoDataFrame
    :return: GeoDataFrame with merged data, including population estimates at each address.

    """
    mapping = CatastroINEMapper()

    # Step 1: Initial join by 'within'
    address_with_census = gpd.sjoin(
        cadastral_addresses_with_buildings.to_crs(census_data.crs),
        census_data,
        how="left",
        predicate="within",
        rsuffix="_census",
    )

    # Step 2: Identify mismatches using mismatced municipality codes
    mismatch_addresses = (
            address_with_census["catastro_municipality_code_building"].astype(str)
            != address_with_census["ine_municipality_code"].astype(str).apply(mapping.ine_to_catastro_code)
    )

    # Step 3: Separate matched and mismatched entries
    matched = address_with_census[~mismatch_addresses]
    mismatched = address_with_census[mismatch_addresses].copy()

    redone_matches = []

    # Step 4: Correct mismatches by finding the nearest census unit exluding the incorreclty matched one
    if not mismatched.empty:
        for idx, row in mismatched.iterrows():
            single_building = gpd.GeoDataFrame(
                [row],
                columns=mismatched.columns[:43],
                geometry=[row.geometry_address],  # explicitly pass the geometry
                crs=mismatched.crs
            )
            # Exclude the previously matched census unit (to avoid reassigning to the same)
            filtered_census = census_data[census_data["CUSEC"] != row["CUSEC"]]

            # Run sjoin_nearest to find the new best match
            corrected = gpd.sjoin_nearest(single_building, filtered_census, how="left")

            # Append to the list
            redone_matches.append(corrected)

        # Combine all corrected matches into one GeoDataFrame
        redone_matches_gdf = pd.concat(redone_matches, ignore_index=True)


        # Step 7: Combine the fixed mismatches and previously correct matches
        final_address_with_census = pd.concat([matched, redone_matches_gdf], ignore_index=True)
        final_address_with_census = gpd.GeoDataFrame(
            final_address_with_census,
            geometry="geometry_address",
            crs=census_data.crs

        )
        final_address_with_census = final_address_with_census.drop(columns=["geometry","index_right"])

    else:
        return matched

    return final_address_with_census

def add_population_per_entrance_based_on_dwellings(geodataframe):
    """
    Adds a new column to the GeoDataFrame that estimates the population per entrance based on the number of dwellings.

    :param geodataframe: GeoDataFrame containing the data.
    :type geodataframe: gpd.GeoDataFrame
    :return: GeoDataFrame with the new column added.
    """

    geodataframe["dwellings_per_entrance"] = (geodataframe["numberOfDwellings_building"]/geodataframe["entrance_count_per_address"])

    total_dwellings_per_cusec = (
        geodataframe.groupby("CUSEC")["dwellings_per_entrance"]
        .sum()
        .reset_index(name="total_dwellings_in_cusec")
    )

    geodataframe = geodataframe.merge(total_dwellings_per_cusec, on="CUSEC", how="left")

    geodataframe["population_per_entrance"] = geodataframe["Total"] / geodataframe["total_dwellings_in_cusec"] * geodataframe["dwellings_per_entrance"]

    return geodataframe


def perform_weighted_kmeans(
    gdf: gpd.GeoDataFrame,
    weight_col: str,
    n_clusters: int,
    geometry_col: str = "geometry_address",
    random_state: int = 42
):
    """
    Perform weighted KMeans clustering based on geometry and a weight column.

    :param gdf: GeoDataFrame with geometry and weights.
    :param weight_col: Column name with weights (e.g., 'population_per_entrance').
    :param n_clusters: Number of clusters to generate.
    :param geometry_col: Name of the geometry column to use for coordinates.
    :param random_state: Seed for reproducibility.
    :return: Tuple of:
             - GeoDataFrame with a new 'cluster' column,
             - GeoDataFrame of cluster centroids with 'cluster_id' and geometry.

    """
    gdf_clean = gdf.dropna(subset=[geometry_col, weight_col]).copy()

    coords = np.array([[geom.x, geom.y] for geom in gdf_clean[geometry_col]])

    weights = gdf_clean[weight_col].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(coords, sample_weight=weights)

    gdf_clean["cluster"] = labels

    centroids_coords = kmeans.cluster_centers_
    centroids_geom = [Point(xy) for xy in centroids_coords]
    centroids_gdf = gpd.GeoDataFrame(
        {"cluster_id": np.arange(n_clusters)},
        geometry=centroids_geom,
        crs=gdf.crs
    )

    return gdf_clean, centroids_gdf
