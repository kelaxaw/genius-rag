-- Artist pairs by number of shared songs. Each pair once, ordered by credit position
-- (primary first) via a self-join on song_artists.

SELECT x.name AS primary_artist, y.name AS featured_artist, COUNT(*) AS songs
FROM song_artists a
JOIN song_artists b ON a.song_id = b.song_id AND a.position < b.position
JOIN artists x ON x.id = a.artist_id
JOIN artists y ON y.id = b.artist_id
GROUP BY x.name, y.name
ORDER BY songs DESC;
