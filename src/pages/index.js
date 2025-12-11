import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();

  // Robot SVG Logo
  const RobotLogo = () => (
    <svg
      className={styles.robotLogo}
      viewBox="0 0 100 100"
      width="100"
      height="100"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Robot head */}
      <rect x="25" y="10" width="50" height="45" rx="8" fill="var(--ifm-color-primary)" />

      {/* Robot eyes */}
      <circle cx="38" cy="28" r="5" fill="white" />
      <circle cx="62" cy="28" r="5" fill="white" />
      <circle cx="38" cy="28" r="2" fill="var(--ifm-color-primary)" />
      <circle cx="62" cy="28" r="2" fill="var(--ifm-color-primary)" />

      {/* Robot mouth/LED display */}
      <rect x="40" y="42" width="20" height="4" rx="2" fill="white" />

      {/* Robot body */}
      <rect x="30" y="58" width="40" height="35" rx="5" fill="var(--ifm-color-primary)" />

      {/* Robot arms */}
      <rect x="15" y="62" width="15" height="6" rx="3" fill="var(--ifm-color-primary)" />
      <rect x="70" y="62" width="15" height="6" rx="3" fill="var(--ifm-color-primary)" />

      {/* Robot legs */}
      <rect x="38" y="95" width="8" height="12" rx="2" fill="var(--ifm-color-primary)" />
      <rect x="54" y="95" width="8" height="12" rx="2" fill="var(--ifm-color-primary)" />

      {/* Decorative elements */}
      <circle cx="50" cy="15" r="3" fill="white" opacity="0.8" />
      <rect x="47" y="20" width="6" height="3" fill="white" opacity="0.8" />

      {/* Circuit pattern */}
      <path d="M45 50 L55 50" stroke="white" strokeWidth="0.5" opacity="0.6" />
      <circle cx="40" cy="65" r="1" fill="white" opacity="0.6" />
      <circle cx="60" cy="65" r="1" fill="white" opacity="0.6" />
    </svg>
  );

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.logoContainer}>
          <RobotLogo />
        </div>
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/intro">
            Read the Book 📚
          </Link>
        </div>
      </div>
    </header>
  );
}

function BookOverview() {
  return (
    <section className={styles.bookOverview}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <h2>Physical AI & Humanoid Robotics</h2>
            <p className={styles.overviewText}>
              This comprehensive guide explores the cutting-edge intersection of artificial intelligence and humanoid robotics.
              From fundamental ROS 2 concepts to advanced Vision-Language-Action integration, this book provides everything
              you need to understand and build sophisticated humanoid robots.
            </p>
            <div className={styles.keyFeatures}>
              <div className={styles.featureItem}>
                <h4>🤖 Complete ROS 2 Foundation</h4>
                <p>Master the Robot Operating System 2 framework with practical examples</p>
              </div>
              <div className={styles.featureItem}>
                <h4>🎮 Advanced Simulation</h4>
                <p>Learn Gazebo and Unity integration for realistic robot testing</p>
              </div>
              <div className={styles.featureItem}>
                <h4>🧠 AI Navigation</h4>
                <p>Implement intelligent path planning and localization algorithms</p>
              </div>
              <div className={styles.featureItem}>
                <h4>💬 VLA Integration</h4>
                <p>Combine vision, language, and action for embodied intelligence</p>
              </div>
            </div>
          </div>
          <div className="col col--6">
            <div className={styles.bookPreview}>
              <div className={styles.bookCover}>
                <div className={styles.coverDesign}>
                  <div className={styles.robotSilhouette}></div>
                  <div className={styles.circuitPattern}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function KeyTopics() {
  return (
    <section className={styles.keyTopics}>
      <div className="container">
        <h2 className={styles.sectionTitle}>Key Topics & Chapters</h2>
        <div className="row">
          <div className="col col--4">
            <div className={styles.topicCard}>
              <div className={styles.topicIcon}>🔧</div>
              <h3>ROS 2 Fundamentals</h3>
              <ul>
                <li>Architecture & Concepts</li>
                <li>Nodes & Communication</li>
                <li>Python Integration</li>
                <li>URDF Basics</li>
              </ul>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.topicCard}>
              <div className={styles.topicIcon}>🎮</div>
              <h3>Simulation & AI</h3>
              <ul>
                <li>Gazebo Setup</li>
                <li>Unity Integration</li>
                <li>Physics Concepts</li>
                <li>Sensor Configuration</li>
              </ul>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.topicCard}>
              <div className={styles.topicIcon}>🧠</div>
              <h3>AI Navigation</h3>
              <ul>
                <li>Path Planning</li>
                <li>Localization</li>
                <li>Mapping</li>
                <li>Control Systems</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col col--4">
            <div className={styles.topicCard}>
              <div className={styles.topicIcon}>💬</div>
              <h3>Vision-Language-Action</h3>
              <ul>
                <li>Vision Systems</li>
                <li>Language Understanding</li>
                <li>Action Planning</li>
                <li>Integration Challenges</li>
              </ul>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.topicCard}>
              <div className={styles.topicIcon}>🏗️</div>
              <h3>Capstone Project</h3>
              <ul>
                <li>System Integration</li>
                <li>Testing Strategies</li>
                <li>Troubleshooting</li>
                <li>Best Practices</li>
              </ul>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.topicCard}>
              <div className={styles.topicIcon}>🔬</div>
              <h3>Advanced Concepts</h3>
              <ul>
                <li>Embodied Intelligence</li>
                <li>Physical AI</li>
                <li>Human-Robot Interaction</li>
                <li>Future Directions</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function AuthorSection() {
  return (
    <section className={styles.authorSection}>
      <div className="container">
        <div className="row">
          <div className="col col--4">
            <div className={styles.authorImage}>
              <div className={styles.authorPlaceholder}></div>
            </div>
          </div>
          <div className="col col--8">
            <h2>About the Author</h2>
            <p>
              A leading expert in humanoid robotics and artificial intelligence with extensive experience
              in developing embodied intelligence systems. The author has contributed to numerous research
              projects and publications in the field of robotics and AI.
            </p>
            <p>
              With a background in computer science and robotics engineering, the author brings both
              theoretical knowledge and practical implementation experience to this comprehensive guide.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Testimonials() {
  return (
    <section className={styles.testimonials}>
      <div className="container">
        <h2 className={styles.sectionTitle}>What Readers Say</h2>
        <div className="row">
          <div className="col col--4">
            <div className={styles.testimonialCard}>
              <p>"An excellent resource for anyone interested in humanoid robotics. The practical examples and clear explanations make complex concepts accessible."</p>
              <div className={styles.testimonialAuthor}>- Robotics Researcher</div>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.testimonialCard}>
              <p>"Finally, a comprehensive guide that bridges the gap between theory and practice in humanoid robotics. Highly recommended!"</p>
              <div className={styles.testimonialAuthor}>- AI Engineer</div>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.testimonialCard}>
              <p>"The step-by-step approach and real-world examples helped me understand complex AI navigation concepts that I struggled with before."</p>
              <div className={styles.testimonialAuthor}>- Robotics Student</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2>Start Your Journey in Humanoid Robotics</h2>
            <p>Access the complete book and begin building intelligent humanoid robots today</p>
            <div className={styles.ctaButtons}>
              <Link
                className="button button--primary button--lg"
                to="/intro">
                Read Online
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/intro">
                Download PDF
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Physical AI & Humanoid Robotics Book`}
      description="Complete guide to humanoid robotics: From ROS 2 fundamentals to advanced Vision-Language-Action integration">
      <HomepageHeader />
      <main>
        <BookOverview />
        <KeyTopics />
        <AuthorSection />
        <Testimonials />
        <CTASection />
      </main>
    </Layout>
  );
}