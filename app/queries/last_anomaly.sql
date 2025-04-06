WITH LatestAnomalyPerTrip AS (
    SELECT
        ad.trip_id,
        MAX(ad.usavp_id) AS last_usavp_id,
        MAX(ad.ml_id) AS last_ml_id
    FROM
        "AnomalyData" as ad
    GROUP BY
        ad.trip_id
), USAVPAnomalyCoordinates AS (
    SELECT
        lad.trip_id,
        u.kilometer_value + u.picket_value * 0.1 AS last_usavp_coordinate
    FROM
        LatestAnomalyPerTrip as lad
    JOIN
        "USAVPdata" as u ON u.usavp_id = lad.last_usavp_id
), MLAnomalyCoordinates AS (
    SELECT
        lad.trip_id,
        m.kilometer_value + m.picket_value * 0.1 AS last_ml_coordinate
    FROM
        LatestAnomalyPerTrip lad
    JOIN
        "MLData" as m ON m.ml_id = lad.last_ml_id
)
SELECT
    lad.trip_id,
    uac.last_usavp_coordinate,
    mac.last_ml_coordinate
FROM
    LatestAnomalyPerTrip as lad
LEFT JOIN
    USAVPAnomalyCoordinates as uac ON lad.trip_id = uac.trip_id
LEFT JOIN
    MLAnomalyCoordinates as mac ON lad.trip_id = mac.trip_id;