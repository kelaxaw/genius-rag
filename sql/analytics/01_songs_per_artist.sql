-- Songs per artist, descending. Shared songs count for every credited artist.

SELECT artists.name, COUNT(*) as songs
FROM song_artists
JOIN artists ON artists.id = song_artists.artist_id
GROUP BY artists.name
ORDER BY songs DESC
LIMIT 15;
