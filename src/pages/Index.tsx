
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { toast } from '@/hooks/use-toast';
import { TrafficRuleCard } from '@/components/TrafficRuleCard';
import { ShareButtons } from '@/components/ShareButtons';
import { LeaderBoard } from '@/components/LeaderBoard';
import { CounterDisplay } from '@/components/CounterDisplay';

const Index = () => {
  const [personalCount, setPersonalCount] = useState(0);
  const [globalCount, setGlobalCount] = useState(4827);
  const [canIncrement, setCanIncrement] = useState(true);
  const [lastIncrement, setLastIncrement] = useState<number | null>(null);

  useEffect(() => {
    // Load personal count from localStorage
    const savedCount = localStorage.getItem('daily-yield-count');
    const savedDate = localStorage.getItem('daily-yield-date');
    const today = new Date().toDateString();
    
    if (savedDate === today && savedCount) {
      setPersonalCount(parseInt(savedCount));
    } else {
      // Reset for new day
      localStorage.setItem('daily-yield-date', today);
      localStorage.setItem('daily-yield-count', '0');
      setPersonalCount(0);
    }

    // Check if user can increment (10 second cooldown)
    const lastIncrementTime = localStorage.getItem('last-increment-time');
    if (lastIncrementTime) {
      const timeDiff = Date.now() - parseInt(lastIncrementTime);
      if (timeDiff < 10000) {
        setCanIncrement(false);
        setTimeout(() => setCanIncrement(true), 10000 - timeDiff);
      }
    }
  }, []);

  const handleIncrement = () => {
    if (!canIncrement) {
      toast({
        title: "Yavaş ol dostum! 🛑",
        description: "10 saniye bekle, sonra tekrar dene.",
        duration: 3000,
      });
      return;
    }

    if (personalCount >= 200) {
      toast({
        title: "Yeter artık! 😅",
        description: "Günlük limiti aştın. Yarın tekrar gel!",
        duration: 3000,
      });
      return;
    }

    const newCount = personalCount + 1;
    setPersonalCount(newCount);
    setGlobalCount(prev => prev + 1);
    setCanIncrement(false);
    setLastIncrement(Date.now());

    // Save to localStorage
    localStorage.setItem('daily-yield-count', newCount.toString());
    localStorage.setItem('last-increment-time', Date.now().toString());

    // Easter egg for 42
    if (newCount === 42) {
      toast({
        title: "Galiba bütün şehir sende direksiyon bırakıyor 🤯",
        description: "42 kişiye yol vermek... Hayat evreni ve her şeyin cevabı!",
        duration: 5000,
      });
    }

    // Re-enable after 10 seconds
    setTimeout(() => setCanIncrement(true), 10000);

    toast({
      title: "Aferin! 👏",
      description: `Bugün ${newCount} kişiye yol verdin!`,
      duration: 2000,
    });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section 
        className="relative min-h-[70vh] bg-cover bg-center bg-no-repeat flex items-center justify-center"
        style={{
          backgroundImage: "linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1920&q=80')"
        }}
      >
        <div className="text-center text-white px-4 max-w-2xl">
          <h1 className="text-4xl md:text-6xl font-bold mb-4 text-yellow-400">
            Yaya gördün mü? Frene bas! 🚶‍♀️
          </h1>
          <p className="text-xl md:text-2xl mb-8 text-gray-100">
            Şimdi say bakalım kaç kişiye yol verdin.
          </p>
          
          {/* Personal Counter */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 mb-6">
            <CounterDisplay count={personalCount} label="Bugün verdiğin yol sayısı" />
          </div>

          <Button 
            onClick={handleIncrement}
            disabled={!canIncrement}
            className="bg-yellow-400 hover:bg-yellow-500 text-black text-xl px-8 py-4 h-auto font-bold rounded-full shadow-lg transform transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Bir yayaya yol verdim düğmesi"
          >
            {canIncrement ? "+1 verdim 🖐️" : "Bekle... ⏰"}
          </Button>
          
          {lastIncrement && !canIncrement && (
            <p className="text-sm text-yellow-200 mt-2">
              10 saniyede bir sayabilirsin!
            </p>
          )}
        </div>
      </section>

      {/* Global Counter */}
      <section className="py-12 bg-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">
            Toplam Gönüllü Yol Veren 🚦
          </h2>
          <CounterDisplay count={globalCount} label="Bugün toplam" size="large" />
        </div>
      </section>

      {/* Traffic Rule Cards */}
      <section className="py-12 bg-gray-50">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">
            Hatırlatma: Trafik Kuralları 📚
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <TrafficRuleCard
              title="Yaya geçidinde dur!"
              description="Art. 74, Karayolları Trafik Kanunu"
              emoji="🛑"
              detail="Yaya geçidinde bekleyen pedestrians have right of way!"
            />
            <TrafficRuleCard
              title="30 km/s yavaşla!"
              description="Özellikle okul yakınlarında"
              emoji="🏫"
              detail="Çocuklar her an yola çıkabilir. Hazırlıklı ol!"
            />
            <TrafficRuleCard
              title="Korna ≠ süper güç"
              description="Sabır en büyük erdem"
              emoji="🔇"
              detail="Korna çalmak sorunu çözmez, gürültü yapar!"
            />
          </div>
        </div>
      </section>

      {/* Share Section */}
      <section className="py-12 bg-yellow-400">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold mb-6 text-black">
            Paylaş ve Övün! 📱
          </h2>
          <p className="text-lg mb-6 text-gray-800">
            Bugün {personalCount} kişiye yol verdiğini herkese duyur!
          </p>
          <ShareButtons count={personalCount} />
        </div>
      </section>

      {/* Leaderboard */}
      <section className="py-12 bg-white">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-8 text-gray-800">
            Şeref Tablosu 🏆
          </h2>
          <LeaderBoard />
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-lg mb-4">
            Bu site mizah içerir, ama mesaj ciddidir. 🚶‍♀️❤️
          </p>
          <p className="text-sm text-gray-400">
            © 2025 Pedestrian Ninja Squad | 
            <span className="mx-2">•</span>
            <a href="#" className="hover:text-yellow-400">Gizlilik</a>
            <span className="mx-2">•</span>
            <a href="#" className="hover:text-yellow-400">İletişim</a>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
