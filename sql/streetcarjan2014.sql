-- SQL (suitable for Postgres) to create table that can be loaded with "Streetcar Jan 2014.csv"

CREATE TABLE public.streetcarjan2014
(
    "Report Date" character varying(10),
    "Route" character varying(10),
    "Time" timestamp without time zone,
    "Day" character varying(10),
    "Location " character varying(200),
    "Incident" character varying(80),
    "Min Delay" integer,
    "Min Gap " integer,
    "Direction" character varying(6),
    "Vehicle" character varying(10)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.streetcarjan2014
    OWNER to postgres;