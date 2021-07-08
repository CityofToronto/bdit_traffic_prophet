-- DROP MATERIALIZED VIEW prj_volume.tp_centreline_lonlat;

CREATE MATERIALIZED VIEW prj_volume.tp_centreline_lonlat
TABLESPACE pg_default
AS
 SELECT geo_id AS centreline_id,
    fcode_desc,
    geom,
    st_x(st_lineinterpolatepoint(st_linemerge(geom), 0.5::double precision)) AS lon,
    st_y(st_lineinterpolatepoint(st_linemerge(geom), 0.5::double precision)) AS lat
   FROM gis.centreline
  WHERE NOT (fcode_desc::text = ANY (ARRAY['River'::character varying, 'Major Shoreline'::character varying, 'Minor Shoreline (Land locked)'::character varying, 'Ferry Route'::character varying, 'Major Railway'::character varying, 'Pending'::character varying, 'Geostatistical line'::character varying, 'Other'::character varying, 'Walkway'::character varying, 'Trail'::character varying, 'Minor Railway'::character varying, 'Hydro Line'::character varying, 'Creek/Tributary'::character varying]::text[]))
WITH DATA;

ALTER TABLE prj_volume.tp_centreline_lonlat
    OWNER TO prj_volume_admins;

COMMENT ON MATERIALIZED VIEW prj_volume.tp_centreline_lonlat
    IS 'Lon-lat centres of centreline segments, for TEPS and Traffic Prophet.';

GRANT SELECT ON TABLE prj_volume.tp_centreline_lonlat TO bdit_humans;