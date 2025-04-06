CREATE TABLE "Trains" (
  "train_id" integer PRIMARY KEY,
  "type" integer,
  "start_exp_date" timestamp
);

CREATE TABLE "Trips" (
  "trip_id" integer PRIMARY KEY,
  "train_id" integer,
  "route" varchar,
  "start_time" timestamp,
  "finish_time" timestamp
);

CREATE TABLE "USAVPdata" (
  "usavp_id" integer PRIMARY KEY,
  "trip_id" integer,
  "kilometer_value" integer,
  "picket_value" integer,
  "time_track" timestamp
);

CREATE TABLE "MLData" (
  "ml_id" integer PRIMARY KEY,
  "trip_id" integer,
  "kilometer_value" integer,
  "picket_value" integer,
  "time_track" timestamp
);

CREATE TABLE "AnomalyData" (
  "anomaly_id" integer PRIMARY KEY,
  "trip_id" integer,
  "usavp_id" integer,
  "ml_id" integer
);

ALTER TABLE "Trips" ADD FOREIGN KEY ("train_id") REFERENCES "Trains" ("train_id");

ALTER TABLE "USAVPdata" ADD FOREIGN KEY ("trip_id") REFERENCES "Trips" ("trip_id");

ALTER TABLE "MLData" ADD FOREIGN KEY ("trip_id") REFERENCES "Trips" ("trip_id");

ALTER TABLE "AnomalyData" ADD FOREIGN KEY ("trip_id") REFERENCES "Trips" ("trip_id");

ALTER TABLE "AnomalyData" ADD FOREIGN KEY ("usavp_id") REFERENCES "USAVPdata" ("usavp_id");

ALTER TABLE "AnomalyData" ADD FOREIGN KEY ("ml_id") REFERENCES "MLData" ("ml_id");
