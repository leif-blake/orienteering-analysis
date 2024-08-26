BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "classes" (
	"id"	INTEGER,
	"name"	TEXT,
	"lower_age"	INTEGER,
	"upper_age"	INTEGER,
	"sex"	TEXT,
    "is_random" INTEGER CHECK (is_random IN (0, 1)),
	PRIMARY KEY("id" AUTOINCREMENT),
	UNIQUE(name)
);
CREATE TABLE IF NOT EXISTS "competitors" (
	"id"	INTEGER,
	"given_name"	TEXT,
	"surname"	TEXT,
	"middle_name"	TEXT,
	"age"	INTEGER,
	"yob"	INTEGER,
	"sex"	INTEGER,
	"club"	TEXT,
	"nationality"	TEXT,
	"person_no" INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT),
	UNIQUE(given_name, surname, club)
);
CREATE TABLE IF NOT EXISTS "courses" (
	"id"	INTEGER,
	"name"	TEXT,
	"number"	INTEGER,
	"length"	NUMERIC,
	"climb"	NUMERIC,
	"controls"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
	UNIQUE(name, controls)
);
CREATE TABLE IF NOT EXISTS "races" (
	"id"	INTEGER,
	"name"	TEXT,
	"date"	INTEGER,
	"location"	TEXT,
	"tz_offset" NUMERIC,
	PRIMARY KEY("id" AUTOINCREMENT),
	UNIQUE(name)
);
CREATE TABLE IF NOT EXISTS "races_competitors" (
	"id"	INTEGER,
	"race_id"	INTEGER,
	"competitor_id"	INTEGER,
	"class_id"	INTEGER,
	"card_no"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY(`race_id`) REFERENCES `races`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY(`competitor_id`) REFERENCES `competitors`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
	UNIQUE(race_id, competitor_id, class_id, card_no)
);
CREATE TABLE IF NOT EXISTS "races_courses_classes" (
	"id"	INTEGER,
	"race_id"	INTEGER,
	"course_id"	INTEGER,
	"class_id"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY(`race_id`) REFERENCES `races`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY(`course_id`) REFERENCES `courses`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
	FOREIGN KEY(`class_id`) REFERENCES `classes`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
	UNIQUE(race_id, class_id)
);
CREATE TABLE IF NOT EXISTS "results" (
	"id"	INTEGER,
	"race_id"	INTEGER,
	"card_no"	TEXT,
	"start_time"	INTEGER,
	"finish_time"	INTEGER,
	"controls"	TEXT,
	"control_times"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY(`race_id`) REFERENCES `races`(`id`) ON DELETE CASCADE ON UPDATE CASCADE
);
COMMIT;
