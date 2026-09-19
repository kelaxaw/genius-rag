-- Annotations per artist: total, per song, and mean text length in characters.
-- count(DISTINCT s.id) for songs: after the joins one row is one annotation.

SELECT
    a.name AS artist,
    COUNT(DISTINCT s.id) AS songs,
    COUNT(*) AS annotations,
    ROUND(COUNT(*)::numeric / COUNT(DISTINCT s.id), 1) AS per_song,
    ROUND(AVG(length(ann.text))) AS avg_len
FROM artists a
JOIN song_artists sa ON a.id = sa.artist_id
JOIN songs s ON sa.song_id = s.id
JOIN annotations ann ON ann.song_id = s.id
GROUP BY a.name
ORDER BY annotations DESC;
