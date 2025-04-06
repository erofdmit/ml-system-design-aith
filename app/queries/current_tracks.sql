WITH RelevantTrips AS (
    SELECT 
        trip_id, route, train_id
    FROM 
        "Trips"
    WHERE 
        start_time IS NOT NULL AND finish_time IS NULL
),
USAVPLastCoordinate AS (
    SELECT 
        trip_id, 
        kilometer_value + picket_value * 0.1 AS last_usavp_coordinate,
        ROW_NUMBER() OVER(PARTITION BY trip_id ORDER BY time_track DESC) AS rn
    FROM 
        "USAVPdata"
    WHERE 
        trip_id IN (SELECT trip_id FROM RelevantTrips)
),
MLLastCoordinate AS (
    SELECT 
        trip_id, 
        kilometer_value + picket_value * 0.1 AS last_ml_coordinate,
        ROW_NUMBER() OVER(PARTITION BY trip_id ORDER BY time_track DESC) AS rn
    FROM 
        "MLData"
    WHERE 
        trip_id IN (SELECT trip_id FROM RelevantTrips)
)
SELECT 
    r.trip_id,
    r.train_id,
    r.route,
    u.last_usavp_coordinate,
    m.last_ml_coordinate
FROM 
    RelevantTrips r
LEFT JOIN 
    USAVPLastCoordinate u ON r.trip_id = u.trip_id AND u.rn = 1
LEFT JOIN 
    MLLastCoordinate m ON r.trip_id = m.trip_id AND m.rn = 1;
