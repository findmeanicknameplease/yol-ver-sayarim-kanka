
import { Button } from '@/components/ui/button';

interface ShareButtonsProps {
  count: number;
}

export const ShareButtons = ({ count }: ShareButtonsProps) => {
  const shareText = `Bugün ${count} kişiye yol verdim! 🚶‍♀️ Sen kaç kişiye verdin? #KacKisiyeVerdin #TrafikNinja`;
  const url = 'https://kackisiyeverdin.com';

  const shareToTwitter = () => {
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(url)}`;
    window.open(twitterUrl, '_blank');
  };

  const shareToWhatsApp = () => {
    const whatsappUrl = `https://wa.me/?text=${encodeURIComponent(shareText + ' ' + url)}`;
    window.open(whatsappUrl, '_blank');
  };

  const shareToInstagram = () => {
    // Copy text to clipboard for Instagram
    navigator.clipboard.writeText(shareText + ' ' + url);
    alert('Metin kopyalandı! Instagram Story\'ne yapıştırabilirsin 📱');
  };

  return (
    <div className="flex flex-wrap justify-center gap-4">
      <Button 
        onClick={shareToTwitter}
        className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-full font-semibold"
      >
        🐦 Twitter'da Paylaş
      </Button>
      <Button 
        onClick={shareToWhatsApp}
        className="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-full font-semibold"
      >
        💬 WhatsApp'ta Paylaş
      </Button>
      <Button 
        onClick={shareToInstagram}
        className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-6 py-3 rounded-full font-semibold"
      >
        📸 Instagram Story
      </Button>
    </div>
  );
};
