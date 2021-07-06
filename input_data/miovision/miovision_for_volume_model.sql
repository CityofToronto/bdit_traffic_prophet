--to find `fnode` and `tnode` matched to a miovision intersection to be used in function
CREATE TABLE jchew.miovision_centreline AS 
    SELECT  X.intersection_uid, X.intersection_name, X.int_id, X.geom AS point,
    Y.geo_id, Y.lf_name, 'from_node' AS node, Y.geom AS centreline
    FROM miovision_api.intersections X
    LEFT JOIN gis.centreline Y ON X.int_id = Y.fnode
UNION
    SELECT  X.intersection_uid, X.intersection_name, X.int_id, X.geom AS point,
    Y.geo_id, Y.lf_name, 'to_node' AS node, Y.geom AS centreline
    FROM miovision_api.intersections X
    LEFT JOIN gis.centreline Y ON X.int_id = Y.tnode
ORDER BY intersection_uid

--delete the one and only outlier
DELETE FROM jchew.miovision_centreline WHERE geo_id = 30097856;

--Then get direction
UPDATE jchew.miovision_centreline SET road_dir = gis.direction_from_line(centreline) WHERE node = 'from_node' ;
UPDATE jchew.miovision_centreline SET road_dir = gis.direction_from_line(ST_Reverse(centreline)) WHERE node = 'to_node' ;

UPDATE jchew.miovision_centreline SET road_dir = 'Northbound' WHERE geo_id = 1145090; --to fix the only wrong one
UPDATE jchew.miovision_centreline SET road_dir = 'N' WHERE road_dir = 'Northbound';
UPDATE jchew.miovision_centreline SET road_dir = 'S' WHERE road_dir = 'Southbound';
UPDATE jchew.miovision_centreline SET road_dir = 'E' WHERE road_dir = 'Eastbound';
UPDATE jchew.miovision_centreline SET road_dir = 'W' WHERE road_dir = 'Westbound';

--Pulling data into the desired format
TRUNCATE TABLE jchew.miovision_volume_model;
INSERT INTO jchew.miovision_volume_model  
SELECT X.volume_15min_uid AS volume_id,
    X.intersection_uid,
    X.datetime_bin AS count_bin,
    X.classification_uid, 
    X.leg, X.dir, X.volume,
    Y.geo_id AS centreline_id,
    CASE WHEN X.dir = 'NB' OR X.dir = 'EB' THEN '1'
    WHEN X.dir = 'SB' OR X.dir = 'WB' THEN '-1'
    END AS dir_bin
FROM miovision_api.volumes_15min X
LEFT JOIN jchew.miovision_centreline Y
ON X.intersection_uid = Y.intersection_uid AND X.leg = Y.road_dir
    WHERE datetime_bin::date BETWEEN '2019-05-01' AND '2019-05-31'  --pulling it month by month from May 2019 onwards
    AND X.classification_uid IN (1,4,5,8)                           --light vehicles & trucks & workvan only
    AND X.intersection_uid IN (1,2,3,5,9,11,18,21,25,26,28,30,31)   --13 selected intersections only
    AND Y.geo_id IS NOT NULL                                        --to filter out some zero bins that were created for invalid movements
ORDER BY X.intersection_uid, datetime_bin 
