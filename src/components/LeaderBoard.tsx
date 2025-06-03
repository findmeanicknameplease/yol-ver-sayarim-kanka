
import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface LeaderboardEntry {
  nickname: string;
  count: number;
  rank: number;
}

export const LeaderBoard = () => {
  const [leaders, setLeaders] = useState<LeaderboardEntry[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    // Mock leaderboard data - in real implementation this would come from Firebase
    const mockLeaders: LeaderboardEntry[] = [
      { nickname: "YayaDostu42", count: 89, rank: 1 },
      { nickname: "TrafikNinja", count: 76, rank: 2 },
      { nickname: "IstanbulShoforu", count: 64, rank: 3 },
      { nickname: "KibarSurucu", count: 58, rank: 4 },
      { nickname: "YolVerenYigin", count: 52, rank: 5 },
      { nickname: "SabirliDragon", count: 47, rank: 6 },
      { nickname: "ZebraGecidi", count: 43, rank: 7 },
      { nickname: "FrenBasam", count: 39, rank: 8 },
      { nickname: "YayaSaygisi", count: 35, rank: 9 },
      { nickname: "TrafikKurali", count: 31, rank: 10 },
    ];
    
    setLeaders(mockLeaders);

    // Rotate through leaders every 3 seconds
    const interval = setInterval(() => {
      setCurrentIndex(prev => (prev + 3) % mockLeaders.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const visibleLeaders = leaders.slice(currentIndex, currentIndex + 3);
  if (visibleLeaders.length < 3 && leaders.length > 0) {
    visibleLeaders.push(...leaders.slice(0, 3 - visibleLeaders.length));
  }

  const getRankEmoji = (rank: number) => {
    switch (rank) {
      case 1: return "🥇";
      case 2: return "🥈";  
      case 3: return "🥉";
      default: return "🏅";
    }
  };

  return (
    <Card className="max-w-2xl mx-auto">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">Günün Kahramanları</CardTitle>
        <p className="text-gray-600">En çok yol veren sürücüler</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {visibleLeaders.map((leader, index) => (
            <div 
              key={`${leader.rank}-${index}`}
              className="flex items-center justify-between p-4 bg-gradient-to-r from-yellow-50 to-yellow-100 rounded-lg border border-yellow-200 transition-all duration-500 hover:shadow-md"
            >
              <div className="flex items-center gap-4">
                <span className="text-2xl">{getRankEmoji(leader.rank)}</span>
                <div>
                  <div className="font-semibold text-gray-800">#{leader.rank}</div>
                  <div className="text-sm text-gray-600">{leader.nickname}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-yellow-600">{leader.count}</div>
                <div className="text-sm text-gray-600">yol verme</div>
              </div>
            </div>
          ))}
        </div>
        <div className="text-center mt-6">
          <p className="text-sm text-gray-500">
            Liste her 5 dakikada güncellenir • Sıralama gerçek zamanlı
          </p>
        </div>
      </CardContent>
    </Card>
  );
};
