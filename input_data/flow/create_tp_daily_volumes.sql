-- DROP MATERIALIZED VIEW prj_volume.tp_centreline_daily_counts;

CREATE MATERIALIZED VIEW prj_volume.tp_centreline_daily_counts
TABLESPACE pg_default
AS
 WITH step_1 AS (
         SELECT volume_id,
            centreline_id,
            dir_bin,
            count_bin,
            volume,
            count_type
           FROM prj_volume.tp_centreline_volumes
          WHERE count_bin::date >= '2006-01-01'::date AND count_type = 1
        ), step_2 AS (
         SELECT volume_id,
            centreline_id,
            dir_bin,
            count_bin,
            date_trunc('HOUR'::text, count_bin) + round((date_part('MINUTE'::text, count_bin) / '15'::numeric::double precision)::numeric, 0)::double precision * '00:15:00'::interval AS count_bin_rounded,
            volume
           FROM step_1
        ), step_3 AS (
         SELECT centreline_id,
            dir_bin,
            count_bin_rounded::date AS count_date,
            avg(volume) AS volume
           FROM step_2
          GROUP BY centreline_id, dir_bin, count_bin_rounded
        )
 SELECT centreline_id,
    dir_bin AS direction,
    date_part('YEAR'::text, count_date) AS count_year,
    count_date,
    '96'::numeric * avg(volume) AS daily_count
   FROM step_3
  GROUP BY centreline_id, dir_bin, count_date
 HAVING count(volume) >= 24
  ORDER BY (date_part('YEAR'::text, count_date)), centreline_id, dir_bin, count_date
WITH DATA;

ALTER TABLE prj_volume.tp_centreline_daily_counts
    OWNER TO prj_volume_admins;

COMMENT ON MATERIALIZED VIEW prj_volume.tp_centreline_daily_counts
    IS 'Aggregation of tp_centreline_volumes data after 2006-01-01 inclusive to daily bins, averaging duplicate bins and excluding any days of incomplete data. For TEPS and Traffic Prophet.';

GRANT SELECT ON TABLE prj_volume.tp_centreline_daily_counts TO bdit_humans;