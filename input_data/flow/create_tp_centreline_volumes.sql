-- DROP MATERIALIZED VIEW prj_volume.tp_centreline_volumes;

CREATE MATERIALIZED VIEW prj_volume.tp_centreline_volumes
TABLESPACE pg_default
AS
 SELECT row_number() OVER (ORDER BY centreline_id, dir_bin, count_type, count_bin) AS volume_id,
    centreline_id,
    dir_bin,
    count_bin,
    sum(volume) AS volume,
    count_type
   FROM prj_volume.centreline_volumes
  GROUP BY centreline_id, dir_bin, count_bin, count_type
WITH DATA;

ALTER TABLE prj_volume.tp_centreline_volumes
    OWNER TO prj_volume_admins;

COMMENT ON MATERIALIZED VIEW prj_volume.tp_centreline_volumes
    IS 'Copy of prj_volume.uoft_centreline_volumes_output, containing volumes from FLOW, for repeated query. For TEPS and Traffic Prophet.';

GRANT SELECT ON TABLE prj_volume.tp_centreline_volumes TO bdit_humans;

CREATE INDEX tp_centreline_volumes_idx
    ON prj_volume.tp_centreline_volumes USING btree
    (centreline_id, count_bin)
    TABLESPACE pg_default;