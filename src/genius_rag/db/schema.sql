-- Metadata schema. Idempotent: safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;  -- chunks.embedding (later stage)

-- Artist is its own entity; ids are generated (the API gives names only).
-- name UNIQUE is the natural key used for upserts.
CREATE TABLE IF NOT EXISTS artists (
    id   integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text    NOT NULL UNIQUE
);

-- Song. id = Genius song_id.
CREATE TABLE IF NOT EXISTS songs (
    id    bigint PRIMARY KEY,
    title text   NOT NULL,
    album text
);

-- Many-to-many song <-> artist; one row per edge.
-- position keeps the JSONL order: 0 = first primary artist, featured artists follow.
CREATE TABLE IF NOT EXISTS song_artists (
    song_id   bigint   NOT NULL REFERENCES songs (id)   ON DELETE CASCADE,
    artist_id integer  NOT NULL REFERENCES artists (id) ON DELETE CASCADE,
    position  smallint NOT NULL DEFAULT 0,
    PRIMARY KEY (song_id, artist_id)
);

-- One-to-many: an annotation belongs to exactly one song.
-- UNIQUE(song_id, fragment, text) makes reloads idempotent and collapses API duplicates.
CREATE TABLE IF NOT EXISTS annotations (
    id       integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    song_id  bigint NOT NULL REFERENCES songs (id) ON DELETE CASCADE,
    fragment text   NOT NULL,
    text     text   NOT NULL CHECK (text <> ''),
    UNIQUE (song_id, fragment, text)
);

CREATE INDEX IF NOT EXISTS annotations_song_id_idx ON annotations (song_id);

-- Child chunks of annotations. parent = annotations.id (search by chunk, cite the parent).
-- embedding is filled at the embedding stage; NULL until computed. Rebuild = DELETE by
-- annotation_id + INSERT (ON CONFLICT cannot drop stale positions when a chunk count shrinks).
CREATE TABLE IF NOT EXISTS chunks (
    id            integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    annotation_id integer  NOT NULL REFERENCES annotations (id) ON DELETE CASCADE,
    position      smallint NOT NULL,
    lang          text     NOT NULL CHECK (lang IN ('ru', 'en')),
    text          text     NOT NULL CHECK (text <> ''),
    tokens        integer  NOT NULL,
    embedding     vector(384),
    UNIQUE (annotation_id, position)
);

CREATE INDEX IF NOT EXISTS chunks_annotation_id_idx ON chunks (annotation_id);
