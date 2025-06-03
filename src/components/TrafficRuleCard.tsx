
import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface TrafficRuleCardProps {
  title: string;
  description: string;
  emoji: string;
  detail: string;
}

export const TrafficRuleCard = ({ title, description, emoji, detail }: TrafficRuleCardProps) => {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div 
      className="perspective-1000 h-48 cursor-pointer"
      onClick={() => setIsFlipped(!isFlipped)}
    >
      <div className={`relative w-full h-full transition-transform duration-700 transform-style-preserve-3d ${isFlipped ? 'rotate-y-180' : ''}`}>
        {/* Front */}
        <Card className="absolute inset-0 backface-hidden border-2 border-yellow-400 hover:shadow-lg transition-shadow">
          <CardContent className="h-full flex flex-col items-center justify-center p-6 text-center">
            <div className="text-4xl mb-4">{emoji}</div>
            <h3 className="text-xl font-bold mb-2 text-gray-800">{title}</h3>
            <p className="text-gray-600">{description}</p>
          </CardContent>
        </Card>
        
        {/* Back */}
        <Card className="absolute inset-0 backface-hidden rotate-y-180 bg-yellow-400 border-2 border-yellow-500">
          <CardContent className="h-full flex flex-col items-center justify-center p-6 text-center">
            <div className="text-4xl mb-4">{emoji}</div>
            <p className="text-gray-800 font-medium">{detail}</p>
            <p className="text-sm text-gray-700 mt-4">Kartı çevirmek için tıkla</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
